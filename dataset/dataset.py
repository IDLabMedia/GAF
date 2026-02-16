
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms 

# SD VAE latent dataset loader
class LatentDataset(Dataset):
    """
    Latent dataset loader
    Expects a .pt file containing a dict with:
        latents: [N, 4, H, W]
        labels: N (optional, defaults to zero)
        vae_scale: sd vae scale (depends on the sd version used)
        class_to_idx: (optional)
        idx_to_class (optional)
    """

    def __init__(self, file_path: str, pin_memory=True):
        obj = torch.load(file_path, map_location='cpu')

        lat = obj["latents"]
        y = obj["labels"]

        assert isinstance(lat, torch.Tensor) and lat.ndim == 4 and lat.shape[1] == 4, \
            f"latent must be a 4D tensor with 4 channels. Got shape {getattr(lat, 'shape', ':/')}"
        assert isinstance(y, torch.Tensor) and y.ndim == 1 and y.dtype == torch.long and y.numel() == lat.shape[0]

        self.latents = lat.contiguous()
        self.labels  = y.contiguous()

        if pin_memory and torch.cuda.is_available():
            self.latents = self.latents.pin_memory()
            self.labels  = self.labels.pin_memory()
            
        self.scale   = float(obj.get("vae_scale", 0.18215)) # sd1.5
        self.class_to_idx = obj.get("class_to_idx", {})
        self.idx_to_class = obj.get("idx_to_class", [])
        self.hr_names     = obj.get("hr_names", {})

        # build per-class indices
        self.class_indices = {}
        for idx, k in enumerate(self.labels.tolist()):
            self.class_indices.setdefault(int(k), []).append(idx)

    def __len__(self): 
        return self.latents.shape[0]
    
    def __getitem__(self, idx): 
        return self.latents[idx], self.labels[idx]

 
class CIFAR10Dataset(Dataset):
    def __init__(self, root, train = True):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
            ])
        )
 
        self.labels = torch.tensor(self.dataset.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
