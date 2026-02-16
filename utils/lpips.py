import torch

import warnings #ibuprofen
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum")

import lpips
loss_fn_alex = lpips.LPIPS(net='alex') 
loss_fn_alex.to("cuda")
loss_fn_vgg = lpips.LPIPS(net='vgg') 
loss_fn_vgg.to("cuda")

def verify_cycle(z_inti, z_final):
    return loss_fn_alex(z_inti, z_final), loss_fn_vgg(z_inti, z_final)  


@torch.no_grad()   
def latent_gaf_lpips(gaf, z1, z2, t_val=1.0, label=None, device=None):
    """
    LPIPS-style distance in latent space using GAF's trunk features.
    """

    z1, z2 = z1.to(device), z2.to(device)
    B = z1.size(0)

    t = torch.full((B,), t_val, device=device, dtype=torch.float32)

    f1 = gaf.trunk(z1, t, label)
    f2 = gaf.trunk(z2, t, label)

    # Case 1: features are (B, C, H, W)
    if f1.dim() == 4:
        # channel-wise normalize
        f1 = torch.nn.functional.normalize(f1, p=2, dim=1)
        f2 = torch.nn.functional.normalize(f2, p=2, dim=1)
        diff = (f1 - f2) ** 2
        return diff.mean()

    # Case 2: features are (B, T, D) tokens
    elif f1.dim() == 3:
        # normalize across feature dimension
        f1 = torch.nn.functional.normalize(f1, p=2, dim=-1)
        f2 = torch.nn.functional.normalize(f2, p=2, dim=-1)
        diff = (f1 - f2) ** 2
        return diff.mean()

    else:
        raise ValueError(f"Unsupported trunk feature shape: {f1.shape}")