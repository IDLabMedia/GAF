import os, math
import torch
import imageio.v2 as iio
from PIL import Image
from torchvision.utils import save_image
from pathlib import Path
import imageio
import numpy as np
from torchvision.utils import make_grid
from utils.vae import decode


def custom_save_image(path: str, img: torch.Tensor):
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    if img.ndim == 4:
        B, C, H, W = img.shape
        nrows = max(1, int(math.isqrt(B)))
        while B % nrows != 0: nrows -= 1
        ncols = B // nrows
        img = (img[: B].view(nrows, ncols, C, H, W)
                 .permute(2, 0, 3, 1, 4)
                 .contiguous()
                 .view(C, nrows * H, ncols * W))
    x = ((img.detach().to(torch.float32).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    iio.imwrite(path, x.permute(1, 2, 0).cpu().numpy())


def giffer(tmp_img):
    gif_frames = []
    for tensor in tmp_img:
        # Same as save_image with normalize=True, value_range=(-1,1)
        tensor = tensor.clone()
        tensor = (tensor - (-1)) / (1 - (-1))  # normalize to 0-1
        tensor = tensor.clamp(0, 1)
        if tensor.dim() == 4:
            tensor = make_grid(tensor, nrow=int(tensor.shape[0]**0.5) or 1, padding=0)

        grid = (tensor * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        gif_frames.append(grid)

    return gif_frames


def save_pure_trajectory_gif(trajectory, vae, cfg, outdir, class_idx, fps=24,format='gif'):
    """
    Save GIF for a single pure class generation.
    """
    
    # Prevent very large gif files for huge trajectories. (Automatic FPS scaling)
    # If video is longer than 20 seconds, speed it up automatically and change format
    if len(trajectory) > 1000 and fps < 30:
        print(f":/ {len(trajectory)} frames is huge. Increasing FPS to 60 to prevent slow-motion, and changing the file the format to mp4.")
        fps = 60
        format = 'mp4'

    traj_dir = outdir / f"traj_pure_{class_idx}"
    traj_dir.mkdir(exist_ok=True)
    
    frames = []
    
    for i, z_step in enumerate(trajectory):
        img = decode(vae, z_step, cfg)
        if img.dim() == 4:
            img = img.squeeze(0)
        img_cpu = img.detach().cpu()
        
        save_image(img_cpu, traj_dir / f"step_{i:03d}.png", normalize=True, value_range=(-1, 1))
        frames.append(img_cpu)
    
    if format=='gif':
        save_gif(frames, outdir, f"traj_pure_{class_idx}.gif", fps=fps)
    else:
        save_mp4(frames, outdir, f"traj_pure_{class_idx}.mp4", fps=fps)

    print(f"Pure class trajectory saved: {outdir}/traj_pure_{class_idx}.{format}")
    
    return frames


def save_all_pure_evolution_gif(pure_traj_registry, vae, cfg, outdir, fps=3.5):
    """
    Save combined GIF showing all pure classes evolving side by side.
    [Pure_1 | Pure_2 | Pure_3 | ...]
    """
    if not pure_traj_registry:
        return
    
    print("Generating combined pure evolution gif...")
    device = next(vae.parameters()).device
    
    classes = list(pure_traj_registry.keys())
    num_steps = len(pure_traj_registry[classes[0]])
    
    combined_frames = []
    
    for step_idx in range(num_steps):
        row_imgs = []
        for c in classes:
            latent = pure_traj_registry[c][step_idx]
            img = decode(vae, latent.to(device), cfg).detach().cpu()
            if img.dim() == 4:
                img = img.squeeze(0)
            row_imgs.append(img)
        
        # Stack horizontally
        frame = torch.cat(row_imgs, dim=-1)
        combined_frames.append(frame)
    
    save_gif(combined_frames, outdir, "all_pure_evolution.gif", fps=fps)
    print(f"Combined pure class evolution saved: {outdir}/all_pure_evolution.gif")


def save_gif(tensor_frames, path, gif_name, fps=5):
    gif_frames = giffer(tensor_frames)

    if len(gif_frames) > 0:
        for _ in range(15): # breathe bih.
            gif_frames.append(gif_frames[-1])

    gif_path = Path(path) / gif_name
    
    imgs = [Image.fromarray(f) for f in gif_frames]
    
    palette_img = imgs[-1].quantize(colors=256, method=Image.Quantize.MEDIANCUT) # Use last frame to build palette (most color info)
    
    quantized = [img.quantize(palette=palette_img) for img in imgs]
    
    quantized[0].save(gif_path, save_all=True, append_images=quantized[1:], duration=int(1000/fps), loop=0)


def save_mp4(tensor_frames, path, mp4_name, fps=60):
    """
    Saves a list of tensors as an MP4 video.
    """
    out_path = Path(path) / mp4_name
        
    print(f"Saving MP4 ({len(tensor_frames)} frames) to {out_path} at {fps} fps...")

    frames_np = []
    
    for img in tensor_frames:
        img = img.detach().cpu()
        
        if img.dim() == 4:
            img = img.squeeze(0)
            
        if img.min() < 0:
            img = (img.clamp(-1, 1) + 1) / 2
            
        # Permute (C, H, W) -> (H, W, C) for video writer
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        
        frames_np.append(img)

    imageio.mimsave(out_path, frames_np, fps=fps, codec='libx264', quality=9, macro_block_size=1)
    
    print(f"MP4 saved successfully.")

def process_and_save_trajectory(trajectory, vae, cfg, mask_frame, pures, outdir, perm_str, nrow=4):
    """
    Loop through trajectory, and create composite grid: Mask | [Pures] | Gen
    """

    traj_dir = outdir / f"traj_{perm_str}"
    traj_dir.mkdir(exist_ok=True)

    mask_cpu = mask_frame.detach().cpu() if mask_frame is not None else None
    pures_cpu = pures.detach().cpu()
    
    B = trajectory[0].shape[0]
    
    all_composite_frames = [[] for _ in range(B)] # Store frames per batch item
    all_raw_gen_frames = [[] for _ in range(B)]

    for i, z_step in enumerate(trajectory):
        grid_img = decode(vae, z_step, cfg)  # (B, 3, H, W)
        
        for b in range(B):
            
            if mask_cpu is not None:
                if mask_cpu.dim() == 4 and mask_cpu.shape[0] > 1:
                    m = mask_cpu[b]
                elif mask_cpu.dim() == 4:
                    m = mask_cpu[0]
                else:
                    m = mask_cpu
            else:
                m = torch.zeros(3, grid_img.shape[2], grid_img.shape[3])

            if pures_cpu.dim() == 4 and pures_cpu.shape[0] > 1:
                p = pures_cpu[b]
            elif pures_cpu.dim() == 4:
                p = pures_cpu[0]
            else:
                p = pures_cpu

            g = grid_img[b].detach().cpu()

            row_tiles = torch.cat([m, p, g], dim=-1)
            
            save_image(row_tiles, traj_dir / f"step_{i:03d}_b{b}.png", normalize=True, value_range=(-1, 1))
            
            all_composite_frames[b].append(row_tiles)
            all_raw_gen_frames[b].append(g)

    for b in range(B):
        save_gif(all_composite_frames[b], outdir, f"traj_{perm_str}_{b}.gif", fps=24)
    
    print(f"Gif sequences saved in {traj_dir}. {B} gifs saved to {outdir}/traj_{perm_str}_*.gif\n")

    return all_raw_gen_frames


def save_evolution_gif(gif_grid_tensors, perms, pure_traj_registry, mask_frame, vae, cfg, outdir):
    """
    Creates a gif showing the evolution of every permutation at once.
    Mask | [Pures (t)] | Mix
    """
    if len(gif_grid_tensors) == 0:
        return

    print("Generating evolution gif for permutations...")
    
    device = next(vae.parameters()).device
    mask_cpu = mask_frame.detach().cpu()
    
    # gif_grid_tensors[perm_idx][batch_idx][step_idx] = tensor
    num_perms = len(gif_grid_tensors)
    B = len(gif_grid_tensors[0])  
    num_steps = len(gif_grid_tensors[0][0])  # Number of steps

    for b in range(B):
        main_gif_frames = []
        
        for step_idx in range(num_steps):
            frame_rows = []

            for perm_idx, perm in enumerate(perms):
                mx = gif_grid_tensors[perm_idx][b][step_idx] # Mix for this batch, this step
                if mx.dim() == 4:
                    mx = mx[0]

                if mask_cpu.dim() == 4 and mask_cpu.shape[0] > 1:
                    m = mask_cpu[b]
                elif mask_cpu.dim() == 4:
                    m = mask_cpu[0]
                else:
                    m = mask_cpu

                pure_batches = []
                for c in perm:
                    if c in pure_traj_registry:
                        latent = pure_traj_registry[c][step_idx]
                        img = decode(vae, latent.to(device), cfg).detach().cpu()
                        if img.dim() == 4:
                            img = img[b] if img.shape[0] > 1 else img[0]
                        pure_batches.append(img)

                row_strip = torch.cat([m] + pure_batches + [mx], dim=-1)
                frame_rows.append(row_strip)

            full_frame = make_grid(frame_rows, nrow=1, padding=4, pad_value=1, normalize=False)
            main_gif_frames.append(full_frame)

        save_gif(main_gif_frames, outdir, f"all_permutations_grid_{b}.gif", fps=3.5)
    
    print(f"Saved {B} evolution gifs to {outdir}/all_permutations_grid_*.gif")


def save_summary_grid(all_mixes, perms, pure_registry, mask_frame, outdir):
    """
    Creates a png grid of the final results.
    Mask | [Pure] | Mix
    """

    print("Generating all permutuation grid...")
    grid_rows = []
    
    mask_cpu = mask_frame.detach().cpu()

    mix_batch = all_mixes[0].detach().cpu()
    if mix_batch.dim() == 3:
        mix_batch = mix_batch.unsqueeze(0)

    batch_size = mix_batch.shape[0]
    for b in range(batch_size):
        grid_rows = []
        for i, perm in enumerate(perms):
            row_tiles = [mask_cpu]
            mix_batch = all_mixes[i].detach().cpu()
            if mix_batch.dim() == 3: 
                mix_batch = mix_batch.unsqueeze(0)


            if mask_cpu.dim() == 4 and mask_cpu.shape[0] > 1:
                m = mask_cpu[b]
            elif mask_cpu.dim() == 4:
                m = mask_cpu[0]
            else:
                m = mask_cpu # 3D

            pure_batches = []
            for c in perm:
                if c in pure_registry:
                    val = pure_registry[c]
                    if isinstance(val, (list, tuple)): 
                        val = val[0]

                    pure_img = val.detach().cpu()
                    
                    if pure_img.dim() == 3:
                        pure_img = pure_img.squeeze(0) # Make it 4D
                    pure_batches.append(pure_img[b] if pure_img.shape[0]>1 else pure_img[0])

            mx = mix_batch[b]
            row_components = torch.cat([m] + pure_batches + [mx], dim=-1)

            grid_rows.append(row_components)

        final_grid = make_grid(grid_rows, nrow=1, padding=4, pad_value=1, normalize=True, value_range=(-1, 1))
        save_image(final_grid, outdir / f"all_permutations_grid_{b}.png")
    print(f"Summary grid saved to {outdir}/all_permutations_grid.png")
