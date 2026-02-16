import torch 
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
from diffusers import AutoencoderKL


def load_vae(cfg):

    device = torch.device(cfg.device)
    try: # local load first
        vae = AutoencoderKL.from_pretrained(cfg.vae_model, torch_dtype=torch.float32, local_files_only=True).to(device)
    except (OSError, EnvironmentError): # fallback
        print(f"model not found locally. Downloading {cfg.vae_model}...")
        vae = AutoencoderKL.from_pretrained(cfg.vae_model, torch_dtype=torch.float32, local_files_only=False).to(device)
        
    try: 
        vae.enable_slicing() 
        vae.enable_tiling()
    except Exception: pass
    vae.eval().requires_grad_(False)
    scale = float(getattr(vae.config, "scaling_factor", cfg.vae_scale))
    
    return vae, scale


def encode(vae, scale, cfg, img_path):

    tfm = T.Compose([
        T.Resize((cfg.image_size, cfg.image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(0.5*torch.ones(3), 0.5*torch.ones(3)),
    ])

    im = Image.open(img_path).convert("RGB")
    input_tensor = tfm(im).unsqueeze(0).to(vae.device)
    img_enc = vae.encode(input_tensor).latent_dist.mean * scale

    return img_enc, input_tensor


def decode(vae, z, cfg):
    with torch.no_grad():
        img = vae.decode(z.to(torch.float32) / cfg.vae_scale).sample
    return img.float().clamp(-1, 1)