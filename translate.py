import torch
import argparse
from pathlib import Path
from torchvision.utils import make_grid
from torchvision.utils import save_image
from transport.algebra import TA
from utils.vae import load_vae, encode
from utils.checkpoint import load_model
from config import load_cfg

def process_generation(args):

    torch.manual_seed(args.seed) 
    
    # basic sanitization and patrol
    assert args.image_size in {256, 512, 32, 64}, f"Unknown model image size. Currently GAF is not trained for {args.image_size} size image."

    if args.data=="afhq":
        assert set([args.source_class]) <= {0,1,2}, "AFHQ only has classes 0 (cat), 1 (dog), 2 (wild)"

    # good to go...
    cfg = load_cfg(args.data, args.image_size, mode='gen', curr_type='sampler')

    if args.mtd=='pair':
        cfg.ckpt = cfg.ckpt_noswap_nores
        cfg.gen_dir = cfg.gen_dir + "_noswap_nores"
    else:
        cfg.gen_dir = cfg.gen_dir + "_full"    
    
    outdir = Path(cfg.gen_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    gaf = load_model(cfg, device)
    
    vae, scale = load_vae(cfg)
    cfg.vae_scale = scale # locked in 
    steps = args.steps or cfg.preview_nfe
    
    assert args.img_path is not None and args.source_class is not None and args.target_class is not None, "please provide the path, source and target classes"

    z_source, img_source = encode(vae, scale, cfg, args.img_path)
    img_target, trajectory = TA.translate(z_source, cfg, gaf, vae, args.source_class, args.target_class, steps, args.solver)
    
    img_row = [img_source.detach().cpu().squeeze(0), img_target.detach().cpu().squeeze(0)]
    png_final = make_grid(img_row, nrow=2, padding=4, pad_value=1)
    save_image(png_final, outdir / f"image_translation_from_{args.source_class}_to_{args.target_class}.png", normalize=True, value_range=(-1, 1))

    trajectory_final = make_grid(trajectory, nrow=trajectory.shape[0]//2, padding=4, pad_value=1)
    save_image(trajectory_final, outdir / f"image_J_K_trajectory_from_{args.source_class}_to_{args.target_class}.png", normalize=True, value_range=(-1, 1))

    print("Done")
    
def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--source_class", type=int, default=None)
    ap.add_argument("--target_class", type=int, default=None)
    ap.add_argument("--img_path", type=str, default=None, help="source image .{png|jpg}.")
    ap.add_argument("--data", type=str, default="afhq", choices=['afhq', 'imagenet']) 
    ap.add_argument("--solver", type=str, default="euler", choices=["euler", "heun", "rk4", "endpoint"])
    ap.add_argument("--mtd", type=str, default="full", choices=["full", "pair"], 
                    help="Training method: full loss or pair only loss (ablation)")
    args = ap.parse_args()

    process_generation(args)
    
if __name__ == "__main__":
    main()