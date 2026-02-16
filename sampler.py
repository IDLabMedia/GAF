import os
import torch
import argparse, math
from pathlib import Path
from torchvision.utils import make_grid
from torchvision.utils import save_image
import itertools
from transport.algebra import TA
from transport.mask import MASK
from utils.vae import load_vae
from utils.vae import decode
from utils.checkpoint import load_model
from config import load_cfg, set_seed
from utils.save_img import save_evolution_gif, save_pure_trajectory_gif, save_all_pure_evolution_gif, save_summary_grid, process_and_save_trajectory

def process_generation(args):

    #set_seed(args.seed) 

    K = len(args.classes)
    if args.list_masks: return MASK.list_available_masks()
    
    # basic sanitization and patrol
    assert not (invalid := set(args.classes).difference(range(1001))), f"Unkown class {invalid}."
    assert args.image_size in {256, 512, 32, 64}, f"Unknown model image size. Currently GAF is not trained for {args.image_size} size image."
    if args.regions: assert len(args.regions) >= 1, f"AFHQ mode Need {K-1} regions for {K} classes. (Last class = rest)"
    if args.regions and args.data!='afhq':
        print("Regions only work with AFHQ model. Exiting... :/")
        return
    if args.data=="afhq":
        assert set(args.classes) <= {0,1,2}, "AFHQ only has classes 0 (cat), 1 (dog), 2 (wild)"
        if args.regions: 
            assert K>=1, f"AFHQ spatial requires atleast 1 class" 
            assert (K-len(args.regions)) < 2, \
            f"{K} classes and {len(args.regions)} region given. Please specify a spatial region for the second class"
    assert args.data!='celeb' or set(args.classes) <={0}, "CELEBA model has 1 class only :/"
    
    if args.regions:
        assert args.mask_type=="afhq" , "Regions are only supported when mask_type is set to 'afhq'."
    
    if args.mask_type=="afhq":
        assert all(x in ["ears", "eyes", "nose", "mouth" ] for x in args.regions), "You provided mask_type='afhq', please provide the correct regions too."
    
    # good to go...
    cfg = load_cfg(args.data, args.image_size, mode='gen', curr_type='sampler')
    outdir = Path(cfg.gen_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(cfg.device)
    
    gaf = load_model(cfg, device)
    
    vae, scale = load_vae(cfg)
    cfg.vae_scale = scale # locked in 
    steps = args.steps or cfg.preview_nfe

    Hlat, Wlat = args.image_size // 8, args.image_size // 8

    z_noise = torch.randn(args.b, cfg.latent_ch, Hlat, Wlat, device=device, dtype=torch.float32)

    nrow = int(math.sqrt(args.b))
    if nrow * nrow != args.b: nrow = args.b

    os.environ["mask_img"]=args.mask_img if args.mask_img is not None else ""
    
    if args.regions:
        hard_masks, mask_names, ignore = MASK.get_semantic_masks(Hlat, Wlat, K, device, args.mask_type, args.regions) 
    else:
        hard_masks, mask_names = MASK.get_semantic_masks(Hlat, Wlat, K, device, args.mask_type)
    
    masks = MASK.soften_masks_bruteforce(hard_masks, sigma=args.sigma)
    if args.data=="afhq":
        mask_viz = MASK.masks_viz_bw(masks, args.image_size, ignore if args.regions else None) if masks  else torch.tensor([]) # Mask visualization
    else:
        mask_viz = MASK.masks_viz_rgb(masks, args.image_size, seed=args.viz_seed) if masks  else torch.tensor([])
    
    print(f"\nClasses: {args.classes}")
    args.mask_type and print(f"Using mask type: {args.mask_type}")
    mask_names and print(f"Regions: {mask_names}")
    
    unique_classes = sorted(list(set(args.classes)))
    
    pure_registry = {}
    pure_traj_registry = {}

    if not args.skip:
        
        for i, c in enumerate(unique_classes):  
            not i and print(f"Generating Anchors: {unique_classes}")

            z, pure_trajectory = TA.generate_pure(z_noise, cfg, gaf, vae, c, steps, args.solver)
            img = decode(vae, z, cfg) if cfg.mode=="latent" else z
            pure_registry[c] = img
            save_image(img, outdir / f"pure_{c}.png", nrow=nrow, normalize=True, value_range=(-1, 1))

            if args.giffer: #gifify
                pure_traj_registry[c] = [t.clone().detach().cpu() for t in pure_trajectory]
                save_pure_trajectory_gif(pure_trajectory, vae, cfg, outdir, c, fps=24)

        if args.giffer and pure_traj_registry:
            save_all_pure_evolution_gif(pure_traj_registry, vae, cfg, outdir, fps=3.5)
    else:
        print("Skipping pure class generation...\n")

    # class permutuation. You could duck or get permuted. Latent Roulette.
    if args.permute:
        perms = list(itertools.permutations(args.classes))
    else:
        perms = [tuple(args.classes)]

    class_weight_map = {c: w for c, w in zip(args.classes, args.weight)} if args.weight is not None else {}

    all_mixes = []

    has_weights = args.weight is not None
    has_mask = args.mask_img is not None or args.mask_type is not None
    has_alpha = args.alpha is not None
    is_pure_mode = not (has_weights or has_mask or has_alpha) or K==1 or args.data=='celeb'

    if is_pure_mode:
        if not args.skip:
            print(f"Pure Class mode - no compisition. saved images to {outdir}") # already saved, exit.
        else:
            print("You are generating nothing in this run. Please select the right args. --h for more.")
    else:  
        method_map = {1:"Pure Class", 
                    2: "Spatial Hybrid" if has_mask else ("Weighted Composition" if has_weights else ("Scalar Alpha Blend" if has_alpha else "Pure Class")), 
                    "multi": f"Spatial Hybrid ({K}-way)"  if has_mask else (f"Weighted Composition ({K}-way)" if has_weights else f"Pure Class")}
        label = method_map.get(K, method_map["multi"])

        gif_grid_tensors = []
        for i, perm in enumerate(perms):
            perm_str = "_".join(map(str, perm))
            print(f"\nGenerating {label} {i+1}/{len(perms)}: {perm_str}")

            if has_mask and has_weights:
                perm_weight = args.weight if args.permute else [class_weight_map.get(c, 0.0) for c in perm]                 
                z, trajectory = TA.generate_spacial_weighted(z_noise, cfg, gaf, vae, perm, masks, perm_weight, steps, args.solver) 
                img_mix = decode(vae, z, cfg) if cfg.mode=="latent" else z
            elif has_mask:
                z, trajectory = TA.generate_spatial(z_noise, cfg, gaf, vae, perm, masks, steps, args.solver)
                img_mix = decode(vae, z, cfg) if cfg.mode=="latent" else z
            elif has_weights:
                perm_weight = args.weight if args.permute else [class_weight_map.get(c, 0.0) for c in perm]                 
                z, trajectory = TA.generate_weighted_blend(z_noise, cfg, gaf, vae, perm, perm_weight, steps, args.solver) 
                img_mix = decode(vae, z, cfg) if cfg.mode=="latent" else z
            elif args.alpha is not None:
                assert K == 2, f"Scalar Alpha Blend only accepts two classes. {K} classes given. v = (1-alpha)*v1 + alpha*v2"
                z, trajectory = TA.generate_scalar_blend(z_noise, cfg, gaf, vae, perm, args.alpha, steps, args.solver)
                img_mix = decode(vae, z, cfg) if cfg.mode=="latent" else z

            # gridify
            all_mixes.append(img_mix)
            mix_cpu = img_mix.detach().cpu()
            mask_cpu = mask_viz.detach().cpu() if mask_viz.numel() > 0 else None
            
            B = mix_cpu.shape[0]
            
            for b in range(B):
                if mask_cpu is not None:
                        
                    if mask_cpu.dim() == 4 and mask_cpu.shape[0]>1:
                        m = mask_cpu[b]
                    elif mask_cpu.dim() == 4:
                        m = mask_cpu[0]
                    else:
                        m = mask_cpu # 3D
                else:
                    m = torch.zeros(3, args.image_size, args.image_size)

                pures_list = []
                for c in perm:
                    p_batch = pure_registry[c]
                    if isinstance(p_batch, (list, tuple)): 
                        p_batch=p_batch[0]
                    p_batch = p_batch.detach().cpu()

                    if p_batch.dim() == 4:
                        p_img = p_batch[b] if p_batch.shape[0] > 1 else p_batch[0]
                    else:
                        p_img = p_batch
                    pures_list.append(p_img )

                mx = mix_cpu[b]

                components = [m] + pures_list + [mx]
                row_strip = torch.cat(components, dim=-1)

                png_final = make_grid(row_strip, nrow=1, padding=4, pad_value=1)
                save_image(png_final, outdir / f"{b}_mix_{perm_str}.png", normalize=True, value_range=(-1, 1))
                            
            if args.giffer and trajectory is not None:
                pures_4d = torch.cat([pure_registry[c].detach().cpu() for c in perm], dim=-1)

                trajectory_frames = process_and_save_trajectory(trajectory, vae, cfg, mask_cpu, pures_4d, outdir, perm_str, nrow)
                gif_grid_tensors.append(trajectory_frames)
        
        save_summary_grid(all_mixes, perms, pure_registry, mask_viz, outdir) if K>1 else ...
        
        if args.giffer and gif_grid_tensors:
            save_evolution_gif(gif_grid_tensors, perms, pure_traj_registry, mask_viz, vae, cfg, outdir)

    print("Done")
    
def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, nargs="+", required=True, help="Class indices. 1 index = pure, 2 = blend, 3 = tri, 4+ = multi")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--data", type=str, default="imagenet", choices=['celeb', 'afhq', 'cifar', 'imagenet']) 
    ap.add_argument("--solver", type=str, default="endpoint", choices=["euler", "heun", "rk4","endpoint"])
    ap.add_argument("--mask_type", type=str, default=None,
                    choices=["custom", "afhq", "horizontal_2", "vertical_2", "radial_2", "diagonal_2", "horizontal", "vertical", "radial", 
                             "diagonal", "quadrant", "afhq", "horizontal_4", "vertical_4", "radial_4", "auto"],
                    help="Mask layout.")
    ap.add_argument("--regions", type=str, nargs="+", default=None, 
                    choices=["ears", "eyes", "nose", "mouth" ], help="AFHQ face regions to assign to classes (rest goes to last class)")
    ap.add_argument("--sigma", type=float, default=1.25, help="Blur strength.")
    ap.add_argument("--skip", action="store_true", help="Skip the pure class generation and only generate the compositions.")
    ap.add_argument("--weight", type=float, nargs="+", default=None, help="Weights for attribute composition: [w1, w2, w3, ...]")
    ap.add_argument("--alpha", type=float, default=None, help="for scalar blend only.")
    ap.add_argument("--mask_img", type=str, default=None, help="custom mask image .{png|jpg}.")
    ap.add_argument("--viz_seed", type=int, default=0)
    ap.add_argument("--giffer", action="store_true", help="Save trajectory steps in gif format.")
    ap.add_argument("--permute", action="store_true", help="permutaute the classes for more generation, + original order")
    ap.add_argument("--list_masks", action="store_true", help="List available mask types and exit")
    args = ap.parse_args()

    process_generation(args)
    
if __name__ == "__main__":
    main()