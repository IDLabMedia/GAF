import os, re, json
from pathlib import Path
from tqdm import tqdm
from PIL import Image 
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from utils.vae import load_vae


def transform(cfg):
    return T.Compose([
        T.Resize((cfg.image_size, cfg.image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def load_hr_names(mapping_path):
    if mapping_path and os.path.isfile(mapping_path):
        try:
            with open(mapping_path, "r") as f:
                d = json.load(f)
            if isinstance(d, dict):
                return {str(k): str(v) for k, v in d.items()}
        except Exception:
            pass
    return {}


def human_name(folder_name, hr_map):
    # Prefer mapping for synsets; otherwise return folder name
    if folder_name in hr_map:
        return hr_map[folder_name]
    return folder_name.replace("_", " ")

def infer_label(path, class_list):
    parts = {x.lower() for x in path.parts}
    for i, class_name in enumerate(class_list):
        if class_name.lower() in parts:
            return i 
    raise RuntimeError(f"Cannot infer label for {path} among classes: {class_list}")


SYNSET_RE = re.compile(r"^n\d{8}$")
def load_loc_synset_mapping(path: str):
    wnids, names = [], {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wnid, desc = line.split(" ", 1)
            wnids.append(wnid)
            names[wnid] = desc
    class_to_idx = {w: i for i, w in enumerate(wnids)}  
    idx_to_class = wnids
    return class_to_idx, idx_to_class, names

SYNSET_RE = re.compile(r"^n\d{8}$")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".JPEG", ".JPG"}


def discover_imagenet_train_images(root, class_to_idx):
    paths, wnids = [], []
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix in IMG_EXTS):
            continue
        wnid = p.parent.name
        if not SYNSET_RE.match(wnid):
            continue
        if wnid not in class_to_idx:
            continue
        paths.append(str(p))
        wnids.append(wnid)
    if not paths:
        raise RuntimeError(f"No train images found under {root}")
    return paths, wnids


# Precompute SD1.5 VAE altent (deterministic)
def precompute_latents_and_labels(cfg):
    print("="*60)
    print("Precomputing deterministic SDXL VAE latents + aligned labels")
    print("="*60)

    data_root = Path(cfg.data_path)  # imagenet/Data/CLS-LOC/train 
    class_to_idx, hr_name = None, {}

    if hasattr(cfg, "loc_synset_mapping") and cfg.loc_synset_mapping: 
        # process data from Imagenet/Data/CLS-LOC/train 
        class_to_idx, _, hr_names = load_loc_synset_mapping(cfg.loc_synset_mapping)
        img_paths, _ = discover_imagenet_train_images(data_root, class_to_idx)
    else:
        # process data from AFHQ, CELEBA
        image_paths = sorted([p for p in data_root.rglob("*") if p.suffix.lower() in IMG_EXTS])

    assert len(image_paths) > 0, f"No images found under {data_root}"
    
    if class_to_idx is not None:
        pass # Imagenet, already processed :)
    else:
        class_list = getattr(cfg, "class_names", [])
        
    print(f"Found {len(img_paths)} images across {len(class_to_idx)} classes")

    vae, scale = load_vae(cfg)
    tfm = transform(cfg)

    all_latents, all_paths, all_labels, all_names = [], [], [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), cfg.encode_batch_size), desc="Encoding"):
            batch_paths = img_paths[i:i+cfg.encode_batch_size]
            imgs, labs, nms, path = [], [], [], []

            for p in batch_paths:
                try:
                    im = Image.open(p).convert("RGB")
                    imgs.append(tfm(im))
                    p_obj = Path(p).parent.name

                    if class_to_idx is not None:
                        label = class_to_idx[p_obj] # imagenet
                    elif class_list:
                        label = infer_label(p_obj, class_list) # afhq
                    else:
                        label = 0 # celeba

                    labs.append(label)
                    nms.append(p_obj)
                    path.append(p)
                except Exception as e:
                    print(f"[skip] {p} ({e})")

            if not imgs: continue

            x = torch.stack(imgs, 0).to(cfg.device, non_blocking=True)
            z = vae.encode(x).latent_dist.mean * scale
            z = z.to(torch.bfloat16)

            all_latents.append(z.cpu())
            all_labels.extend(labs)
            all_names.extend(nms)
            all_paths.extend(path)

    if not all_latents: raise RuntimeError("No valid images encoded.")

    latents = torch.cat(all_latents, dim=0).contiguous()
    y = torch.tensor(all_labels, dtype=torch.long)

    assert latents.ndim == 4 and latents.shape[1] == 4, f"Shape Error:{latents.shape}"
    assert latents.shape[0] == y.numel() == len(all_paths), \
        f"Alignemnt Error: B={latents.shape[0]}, y={y.nume()}, path={len(all_paths)}"
    
    payload = {
        "latents": latents,                
        "labels": y,                       
        "label_names": all_names,          
        "paths": all_paths,                
        "vae_scale": scale,
        "vae_model": cfg.vae_model,
        "image_size": cfg.image_size,
        "num_classes": getattr(cfg, "num_classes", len(torch.unique(y)))
    }

    if class_to_idx is not None:
        payload["class_to_idx"] = class_to_idx # {folder -> idx}
        payload["idx_to_class"] = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]  # [idx -> folder]
        payload["hr_names"] = hr_names
    
    torch.save(payload, cfg.latent_cache)
    print(f"Saved latents -> {cfg.latent_cache} ({os.path.getsize(cfg.latent_cache)/1024**3:.2f} GB)")
    
    torch.cuda.empty_cache()
    return latents.shape[2], latents.shape[3]
