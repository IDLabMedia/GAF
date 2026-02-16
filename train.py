# Generative Anchored Fields: Controlled Data Generation via Emergent Velocity Fields and Transport Algebra (GAF)
# Deressa Wodajo, Hannes Maaren, Peter Lamber, Glenn Van Wallendwal;
# arXiv: https://arxiv.org/abs/2511.22693v1

import torch 
import torch.nn as nn
import re, random
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm
import argparse

from dataset.precompute import load_vae, load_hr_names, human_name
from utils.save_img import custom_save_image
from models.gaf import GAF
from utils.ema import EMA
from transport.algebra import TA
from utils.vae import decode
from utils.loss import gaf_loss, sample_t_uniform
from utils.checkpoint import load_checkpoint, retrunk
from config import load_cfg, save_to_cfg, set_seed
from dataset.loader import data_loader
from utils.opt import get_optim_groups


def collate_latents(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)


def train(cfg, args):
    set_seed(cfg.seed)

    data = args.data
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_to_cfg(cfg, outdir / "config.yaml") # save current run configs

    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, config=vars(cfg), name=cfg.wandb_run_name)
    
    loader, dataset, input_size = data_loader(cfg, data)
    
    dev = torch.device(cfg.device)
    base = GAF(latent_ch=cfg.latent_ch, input_size=input_size, hidden_size=cfg.dit_hidden,
              depth=cfg.dit_depth, num_heads=cfg.dit_heads, patch_size=cfg.patch_size, 
              mlp_ratio=cfg.mlp_ratio, num_classes=cfg.num_classes).to(dev)
    
    if args.retrunk: # Relaod the trunk from a pretrained model (DiT in our case)
        retrunk(base.trunk, cfg.trunk_ckpt, skip_patterns=cfg.skip, freeze=cfg.freeze)

        # Check what's trainable after retrunking
        trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
        total = sum(p.numel() for p in base.parameters())
        print(f"\nTrainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        
    if cfg.channels_last: base = base.to(memory_format=torch.channels_last)

    model = base

    if cfg.compile:
        try: 
            model = torch.compile(base, mode="max-autotune", fullgraph=False, dynamic=False) # model=defualt, if accum>1
        except Exception as e: 
            print(f"[compile] disabled: {e}")

    print(f"K_heads count: {len(base.twins.K)}")
    print(f"Labels in dataset: min={dataset.labels.min()}, max={dataset.labels.max()}")

    param_groups = get_optim_groups(base, cfg.weight_decay)   
        
    try:
        opt = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas)

    min_lr = 1e-6
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-2, end_factor=1.0, total_iters=cfg.warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, cfg.iters - cfg.warmup_steps), eta_min=min_lr
    )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[cfg.warmup_steps]
    )

    ema = None
    if cfg.use_ema:
        ema = EMA(base, decay_warmup = cfg.ema_decay_warm, decay_main=cfg.ema_decay_main, warmup_steps=cfg.ema_decay_steps)

    vae = None
    if cfg.mode == 'latent':
        vae, _ = load_vae(cfg)

        # load human-readable names map
        hr_map = load_hr_names(cfg.class_names_json)
        idx2folder = dataset.idx_to_class

    if data == 'celeb':
        idx2name = ['celeb']
    elif data == 'cifar':
        idx2name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    else:
        idx2name = [human_name(s, hr_map) for s in idx2folder]

    print(f"[train] {data} lr={cfg.lr} betas={cfg.betas} wd={cfg.weight_decay} | EMA fixed={cfg.ema_decay_main}")
    print(f"[train] batch={cfg.batch}")
    print(f"[train] residual targets: res_reg={cfg.res_reg} lam_time={cfg.lam_time}")

    step, start_step = 0, 0

    if args.resume:
        start_step = load_checkpoint(
            args,
            cfg,
            base,
            ema=ema,
            opt=opt,
            sched=scheduler,
            device=dev
        )
        assert cfg.iters>start_step, f"Last checkpoint iteration matches the current run iteration. Please set a higher resume iteration, e.g, (iter: {cfg.iters+1000})"

    base.train()

    step = start_step
    sub_step = 0
    pbar = tqdm(total=cfg.iters, initial=start_step, desc=f"Iters/{cfg.iters}")
    opt.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    enabled = cfg.bf16 and (dev.type == "cuda")

    while step < cfg.iters:
        try: 
            z_x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader) 
            z_x, y = next(data_iter)

        z_x = z_x.to(dev, non_blocking=cfg.non_blocking)
        y   = y.view(-1).long().to(dev, non_blocking=cfg.non_blocking)

        if cfg.channels_last: 
            z_x = z_x.to(memory_format=torch.channels_last)

        batch_size = z_x.size(0)
        z_y = torch.randn_like(z_x, dtype=torch.float32).to(z_x.dtype)

        t = sample_t_uniform(cfg, batch_size, dev)

        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=enabled):
            L_raw, logs = gaf_loss(model, z_x, z_y, t, y, cfg, step)
            loss = L_raw / cfg.accum_steps
            loss.backward()

        if sub_step % cfg.accum_steps == 0:
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(base.parameters(), cfg.grad_clip)

            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
            step += 1; pbar.update(1)

            if ema:
                ema.update(base, step)

            if step % cfg.log_every == 0:
                pbar.set_description(f"Iters {step}/{cfg.iters}")
                pbar.set_postfix(pair=f"{logs['pair']:.4f}",
                                data=f"{logs['data_loss']:.4f}",
                                noise=f"{logs['noise_loss']:.4f}",
                                lambda_res=f"{logs['lambda_res']:.4f}",
                                lambda_time=f"{logs['lambda_time']:.4f}",
                                res=f"{logs['residual_pen']:.4f}",
                                tanti=f"{logs['time_antisym']:.4f}",
                                lr=f"{cfg.lr}",
                                v_mse= f"{logs['v_mse']:.4f}",
                                cos=f"{logs['cos']:.4f}")
                if cfg.use_wandb: wandb.log({**logs, "lr": scheduler.get_last_lr()[0]}, step=step)

            if cfg.preview_every > 0 and (step % cfg.preview_every == 0):
                with torch.no_grad():
                    src = ema.ema_model if ema else base
                    # pick some classes from the current batch for preview
                    uniq_k = torch.unique(y).tolist()
                    random.shuffle(uniq_k)
                    show = uniq_k[:cfg.preview_k_per_iter] if uniq_k else list(range(cfg.preview_k_per_iter))
                    for k_index in show:
                        name = idx2name[int(k_index)] if 0 <= int(k_index) < len(idx2name) else f"class_{int(k_index)}"
                        z_y = torch.randn_like(z_x, dtype=torch.float32).to(z_x.dtype)
                        
                        z, _ = TA.generate_pure(z_y, cfg, src, vae, k_index, cfg.preview_nfe, method="euler")
                        img = decode(vae, z, cfg) if cfg.mode=="latent" else z

                        safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name.strip())
                        custom_save_image(str(outdir / f"iter_{step:07d}_k{int(k_index)}_{safe}.png"), img)

            if cfg.checkpoint_every > 0 and (step % cfg.checkpoint_every == 0):
                ckpt_path = outdir / f"checkpoint_it{step:07d}.pt"
                torch.save({
                    "model": base.state_dict(), 
                    "ema": ema.state_dict() if ema else None,
                    "opt": opt.state_dict(),
                    "sched": scheduler.state_dict(),
                    "step": step,
                    "cfg": vars(cfg)
                    }, ckpt_path)

    # final model save
    final_state = {
        "model": base.state_dict(),
        "ema": (ema.state_dict() if (cfg.use_ema and ema is not None) else None),
        "opt": opt.state_dict(),
        "sched": scheduler.state_dict(),
        "step": step,
        "cfg": vars(cfg)
    }
    torch.save(final_state, Path(cfg.outdir) / "final_state.pt")

    # final previews
    with torch.no_grad():
        src = ema if (cfg.use_ema and (ema is not None)) else base
        sample_ks = [0, 1, 2, 3, 4, 5]
        for k_index in sample_ks:
            name = idx2name[int(k_index)] if 0 <= int(k_index) < len(idx2name) else f"class_{int(k_index)}"
            Hlat, Wlat = cfg.preview_H, cfg.preview_W
            if cfg.mode=="latent":
                Hlat, Wlat = cfg.preview_H // 8, cfg.preview_W // 8
            z_y = torch.randn(cfg.batch, cfg.latent_ch, Hlat, Wlat, device=dev, dtype=torch.float32)
            z, _ = TA.generate_pure(z_y, cfg, src, vae, k_index, cfg.preview_nfe, method="euler")
            img = decode(vae, z, cfg) if cfg.mode=="latent" else z
            safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name.strip())
            custom_save_image(str(outdir / f"samples_final_k{k_index}_{safe}.png"), img)
    print(f"\n\nModel saved in {outdir}.\n")
    print(f"[DONE :)]")


def main():
    parser = argparse.ArgumentParser(description="Train GAF model")
    parser.add_argument("--data", type=str, default="imagenet", choices=['celeb', 'afhq', 'cifar', 'imagenet'])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retrunk", action="store_true", help="Reload the trunk with pretained model")
    args = parser.parse_args()

    # Load from the YAML
    cfg = load_cfg(args.data, args.image_size, curr_type='train')
    
    # Run
    train(cfg, args)


if __name__ == "__main__":
    main()

