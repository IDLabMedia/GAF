import torch
from torch.utils.data import DataLoader
from .dataset import CIFAR10Dataset as cifar, LatentDataset as latent

def collate_fn(batch):
    xs, ys = zip(*batch)
    x_batch = torch.stack(xs, 0)

    if isinstance(ys[0], torch.Tensor): 
        y_batch = torch.stack(ys).long()
    else:
        y_batch = torch.tensor(ys, dtype=torch.long)

    return x_batch, y_batch

def data_loader(cfg, data="cifar"):
    if data not in ['cifar','celeb','imagenet','afhq']:
        raise NotImplementedError("dataloader not implemeted for this dataset. :/")
    
    mode = 'latent'
    if data == 'cifar':
        dataset = cifar(cfg.data_root, train=True)
        N = len(dataset)
        input_size = cfg.image_size
        mode = 'image'
    else:
        dataset = latent(cfg.latent_cache)
        N, C, Hlat, Wlat = dataset.latents.shape
        assert Hlat == Wlat and (Hlat % cfg.patch_size) == 0
        input_size = Hlat

    if (data == 'imagenet') or (data == 'afhq'): # or any dataset with class label as in imgfolder/class/img1..., (but needs latent precomputing first. :/ )
        if getattr(dataset, "idx_to_class", None):
            num_classes_cache = int(len(dataset.idx_to_class))
        else:
            num_classes_cache = int(dataset.labels.max().item()) + 1
        if cfg.num_classes != num_classes_cache:
            print(f"[warn] cfg.num_classes={cfg.num_classes} != cache num_classes={num_classes_cache} -> using cache value")
        
        cfg.num_classes = num_classes_cache
        
        # Sanity: labels must be within [0, num_classes)
        y_min = int(dataset.labels.min().item())
        y_max = int(dataset.labels.max().item())
        
        assert 0 <= y_min and y_max < cfg.num_classes, f"Label range [{y_min},{y_max}] outside num_classes={cfg.num_classes}"
    
    print(f"[dataset] N={N} | {mode} size={input_size}x{input_size} | classes={cfg.num_classes}")

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.workers > 0),
        collate_fn=collate_fn
    )
    if cfg.workers>0:
        loader_kwargs["prefetch_factor"] = 16

    loader = DataLoader(**loader_kwargs)

    return loader, dataset, input_size