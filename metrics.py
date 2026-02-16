"""Stess Test GAF Drift Through Various Cyclic Transports."""

from pathlib import Path
import argparse
import random
import torch
import numpy as np
import imageio.v2 as iio
from torchvision.utils import save_image
from utils.save_img import save_pure_trajectory_gif
from utils.vae import load_vae
from config import load_cfg
from utils.checkpoint import load_model
from transport.cycle import CYCLE


def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20) # 250+ for high-fidelity cyclic transport
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--n_interp", type=int, default=10) # number of columns (how many cats?).
    ap.add_argument("--classes", type=int, nargs="+", required=True, help="Class to ride through")
    ap.add_argument("--solver", type=str, default="rk4", choices=["euler", "heun", "rk4", "endpoint"])
    ap.add_argument("--data", type=str, default="imagenet", choices=['afhq', 'imagenet']) 
    ap.add_argument("--grid_size", type=int, default=7) # for barycentric
    ap.add_argument("--save", action="store_true", help="Save the gif file.")  
    ap.add_argument("--mode", choices=["interpolation","barycentric", "cycle"], default=None)

    #seed = 35
    
    args = ap.parse_args()
    set_seed(args.seed)

    # basic sanitization and patrol
    assert args.mode, "please provide the cycle mode"
    assert args.image_size in {256, 512, 32, 64}, f"Unknown model image size. Currently GAF is not trained for {args.image_size} size image."
    if args.data=="afhq":
        assert set(args.classes) <= {0,1,2,-1}, "Unknown classes given. AFHQ only has classes 0 (cat), 1 (dog), 2 (wild)"

    loop = args.mode not in ['barycentric'] # check if its cyclic transport or not

    assert len(args.classes)==3 if not loop else True, "Barycentric needs exactly 3 K heads"
    assert len(set(args.classes))==3 if not loop else True, "Barycentric needs exactly 3 UNIQUE 3 K heads"

    class_map = {'afhq':list(range(3)), 'celeb':list(range(1)), 'imagenet':list(range(1000)), 'cifar':list(range(10))}
    classes = class_map[args.data] if -1 in args.classes else args.classes
    
    cfg = load_cfg(args.data, args.image_size)
    print(cfg.ckpt)
    
    device = torch.device(cfg.device)
    gaf = load_model(cfg, device)
    vae, scale = load_vae(cfg)
    cfg.vae_scale = scale # locked in 

    imgs, trajectory = CYCLE.ride(args.mode, cfg, gaf, vae, classes, args.steps, args.solver, track=True)
    main_imgs, latent = imgs['img'], imgs['z']
    fname = f"{args.mode}_transport_{'_'.join(map(str, classes[:20]))}_{random.randint(100000, 999999)}.png"
    
    x = (main_imgs.float().clamp(-1, 1) + 1) * 0.5
    out = Path(cfg.gen_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    save_image(x, out / fname, nrow=args.grid_size if args.mode in ['barycentric'] else args.n_interp)
    print(f"Saved cycle grid to {out / fname}\n")

    if args.save: # save the ride
        class_idx_str = "_".join(map(str, classes[:20]))
        save_pure_trajectory_gif(trajectory, vae, cfg, out, class_idx_str, fps=24)

    elif loop: # cyclic
        print('HERE')
        from utils.lpips import verify_cycle, latent_gaf_lpips
        lpips_alex, lpips_vgg = verify_cycle(main_imgs[0], main_imgs[-1])
        first_z = latent[0]
        last_z = latent[-1]
        
        if first_z.dim()==3:
            first_z = first_z.unsqueeze(0)
        if last_z.dim()==3:
            last_z = last_z.unsqueeze(0)
        
        latent_mse = (first_z - last_z).pow(2).mean().item()                
        pixel_mse = torch.mean((main_imgs[0] - main_imgs[-1]) ** 2)
        latent_lpips_val = latent_gaf_lpips(gaf, first_z, last_z, t_val=1.0, label=torch.tensor(classes[0], device=device), device=device)

        print(f"Pixel MSE={pixel_mse.item()}")
        print(f"LPIPS(alex)={lpips_alex.item()}, \nLPIPS(vgg)={lpips_vgg.item()}")
        print(f"Latent MSE: {latent_mse}")
        print(f"Latent LPIPS-GAF: {latent_lpips_val.item()}\n")

    loop and lpips_alex==0.0 and print("Sub Zero: Flawless Victory! - Circlity.")

if __name__ == "__main__":
    main()

