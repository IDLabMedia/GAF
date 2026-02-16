import os
import torch
from pathlib import Path
from models.gaf import GAF
#from models.rflow import RFlow as GAF

def load_checkpoint(args, cfg, model, ema=None, opt=None, sched=None, device="cpu"):
    ckpt_path = Path(cfg.resume_path) #/ "final_state.pt"

    start_step = 0
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}, starting from scratch.")
        return 0

    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Load Model
    model.load_state_dict(ckpt['model'], strict=True)
    print("Model weights loaded. :)")
    
    # Load EMA
    if ema and (ema is not None) and ('ema' in ckpt) and (ckpt['ema'] is not None):
        ema.load_state_dict(ckpt['ema'], strict=True)
        print("EMA weights loaded. :)")

    # Load Optimizer
    if opt is not None and 'opt' in ckpt:
        try:
            opt.load_state_dict(ckpt['opt'])
            print("Optimizer state loaded. :)")
        except Exception as e:
            print(f"Could not load optimizer state: {e}")

    # Load Scheduler
    if sched is not None and 'sched' in ckpt:
        try:
            sched.load_state_dict(ckpt['sched'])
            print("Scheduler state loaded. :)")
        except Exception as e:
            print(f"Could not load scheduler state: {e}")

    # Load Step
    start_step = ckpt.get('step', 0)
    print(f"Resuming from step {start_step}")
    
    return start_step


def load_model(cfg, device):
    
    if not getattr(cfg, 'ckpt', None):
        raise AttributeError(f"Config is missing 'ckpt' key. Please locate the chekpoint in the config.")
    
    ckpt_path = cfg.ckpt
    print('LOADING...',ckpt_path)
    try:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found on disk: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint at: {cfg.ckpt}") from e
    image_size = cfg.image_size if cfg.data=='cifar' else cfg.image_size // 8
    gaf = GAF(
        latent_ch=cfg.latent_ch,
        input_size=image_size,
        hidden_size=cfg.dit_hidden,
        depth=cfg.dit_depth,
        num_heads=cfg.dit_heads,
        num_classes=cfg.num_classes,
        #added for rectifed flow test
        patch_size=cfg.patch_size,
        mlp_ratio=cfg.mlp_ratio,
    ).to(device).eval()

    print("CFG image_size:", getattr(cfg, "image_size", None))
    print("CFG patch_size:", getattr(cfg, "patch_size", None))

    state = ckpt.get("ema") or ckpt["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()} # maybe compiled model?
   
    # Check if weights are loaded correctly
    missing, unexpected = gaf.load_state_dict(state, strict=True)

    print('GAF model loaded...\n')
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing:
        print("First 5 missing:", missing[:5])
    if unexpected:
        print("First 5 unexpected:", unexpected[:5])
    print("\n")
    
    return gaf

def retrunk(model, trunk_ckpt, skip_patterns=None, freeze=False):
    """Retrunk: loads pretrained weights into the trunk."""

    print(skip_patterns)
    print("\nRetrunking...")
    
    if trunk_ckpt is not None:
        skip_patterns = skip_patterns or []
        print(skip_patterns)
        ckpt = torch.load(trunk_ckpt, map_location='cpu')
        if 'model' in ckpt:
            ckpt = ckpt['model']
        if 'ema' in ckpt:
            ckpt = ckpt['ema']
        
        model_state = model.state_dict()
        loaded, skipped = [], []
        loaded_names = set()
        
        for name, param in ckpt.items():
            if any(p in name for p in skip_patterns):
                skipped.append(f"{name} (pattern)")
                continue
            
            if name not in model_state:
                skipped.append(f"{name} (missing)")
                continue
            
            if model_state[name].shape != param.shape:
                if "embedding_table" in name and param.dim() == 2:
                    if param.shape[1] == model_state[name].shape[1]:
                        n = min(model_state[name].shape[0], param.shape[0])
                        model_state[name][:n] = param[:n]
                        loaded.append(f"{name} (partial {n})")
                        loaded_names.add(name)
                        continue
                skipped.append(f"{name} (shape)")
                continue
            
            model_state[name] = param
            loaded.append(name)
            loaded_names.add(name)
        
        model.load_state_dict(model_state)
        
        # Freeze loaded params
        if freeze:
            frozen = 0
            for name, p in model.named_parameters():
                if name in loaded_names:
                    p.requires_grad_(False)
                    frozen += 1
            print(f"Frozen: {frozen}")
        
        print(f"Loaded: {len(loaded)} | Skipped: {len(skipped)}")
        for s in skipped:
            print(f"  skip: {s}")