"""
A General and brute force masks for ImageNet and AFHQ dataset
"""

import os
import torch
import torch.nn.functional as F


class MASK:
    
    REGISTRY = {}
    
    def register_mask(name, reg=REGISTRY):
        def decorator(fn):
            reg[name] = fn
            return fn
        return decorator


    @register_mask("custom")
    @staticmethod
    def mask_custom(H, W, device):
        """ User provided custom image masks.
            The mask regions will be given numeric names: first, second, third, ...
            Splits image into regions based on unique pixel values.
            Assumes 0 (Black) is background and ignores it.
        """
        from PIL import Image
        import numpy as np

        path = os.getenv("mask_img") #getattr(args, 'mask_img', None)
        
        if not path:
            raise ValueError("Mask type is 'custom' but --mask_img was not provided.")

        try:
            img = Image.open(path).convert("L")
            img = img.resize((W, H), resample=Image.NEAREST)
        except FileNotFoundError:
            raise FileNotFoundError(f"Custom mask file not found at: {path}")

        img_tensor = torch.from_numpy(np.array(img)).to(device)

        unique_vals = torch.unique(img_tensor)

        masks = []
        names = []
        ordinals = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth']
        
        for i, val in enumerate(unique_vals):
            # Create Binary Mask: 1.0 where pixels match val, 0.0 elsewhere
            # Reshape to (1, 1, H, W)
            m = (img_tensor == val).float().view(1, 1, H, W)
            masks.append(m)
            
            # Assign Name
            names.append(ordinals[i] if i < len(ordinals) else f"region_{i+1}")

        print(f"loaded custom mask: {path} with {len(masks)} regions.")
        return masks, names


    @staticmethod
    def get_afhq_face_regions(H, W, device):
        """4 face region bands: eye/mouth/nose/ear"""

        ears = torch.zeros(1, 1, H, W, device=device)
        ears[:, :, :int(H*0.3), :int(W*0.3)] = 1.0 
        ears[:, :, :int(H*0.3), -int(W*0.3):] = 1.0 
        
        eyes = torch.zeros(1, 1, H, W, device=device)
        eyes[:, :, int(H*0.2):int(H*0.45), int(W*0.15):int(W*0.85)] = 1.0
        
        nose = torch.zeros(1, 1, H, W, device=device)
        nose[:, :, int(H*0.4):int(H*0.65), int(W*0.3):int(W*0.7)] = 1.0
        
        mouth = torch.zeros(1, 1, H, W, device=device)
        mouth[:, :, int(H*0.65):int(H*0.85), int(W*0.25):int(W*0.75)] = 1.0
        
        return {"ears": ears, "eyes": eyes, "nose": nose, "mouth": mouth}


    @register_mask("afhq")
    @staticmethod
    def mask_afhq(H, W, device, regions=("eyes", "mouth")):
        """Returns ONLY two features. The rest is calculated in the main logic."""

        boxes = MASK.get_afhq_face_regions(H, W, device)
        if len(regions)==1:
            m1 = boxes[regions[0]]
            return [m1], [regions[0]]
        if len(regions)==2:
            m1 = boxes[regions[0]]
            m2 = boxes[regions[1]]
            return [m1, m2], [regions[0], regions[1]]
        
        if len(regions)==3:
            m1 = boxes[regions[0]]
            m2 = boxes[regions[1]]
            m3 = boxes[regions[2]]
            return [m1, m2, m3], [regions[0], regions[1], regions[2]]


    @staticmethod
    def get_afhq_semantic_masks(H, W, device, regions=("eyes", "mouth")):
        """Compute two specific regions from the registry and get the remaining region. mask3 = 1 - (mask1 + mask2)"""

        masks, names = MASK.REGISTRY["afhq"](H, W, device, regions=regions)

        if len(masks)==1:
            mask1 = masks[0]
            name1 = names[0]
            mask2 = torch.clamp(1.0 - (mask1), 0, 1)
            return [mask1, mask2], [name1, "rest"], True # ignore the last mask (appears blank)
        
        if len(masks)==2:
            mask1, mask2 = masks[0], masks[1]
            name1, name2 = names[0], names[1]
            mask3 = torch.clamp(1.0 - (mask1 + mask2), 0, 1)
            return [mask1, mask2, mask3], [name1, name2, "rest"], True # ignore the last mask (appears blank)
        
        if len(masks) ==3 and len(regions)==3:
            mask1, mask2, mask3 = masks[0], masks[1], masks[2]
            name1, name2, name3 = names[0], names[1], names[2]
        
            return [mask1, mask2, mask3], [name1, name2, name3], False # do not ignore the last mask (we are generating only specific regions)


    # IMAGENET MASKS ALL THE WAY DOWN?, Yup.
    @register_mask("horizontal_2")
    @staticmethod
    def mask_horizontal(H, W, device):
        """2 horizontal bands: top/bottom"""

        masks = []
        for i in range(2):
            m = torch.zeros(1, 1, H, W, device=device)
            m[:, :, int(H * i / 2):int(H * (i + 1) / 2), :] = 1.0
            masks.append(m)
        return masks, ['top', 'bottom']
    

    @register_mask("vertical_2")
    @staticmethod
    def mask_vertical(H, W, device):
        """2 vertical bands: left/right"""

        masks = []
        for i in range(2):
            m = torch.zeros(1, 1, H, W, device=device)
            m[:, :, :, int(W * i / 2):int(W * (i + 1) / 2)] = 1.0
            masks.append(m)
        return masks, ['left', 'right']
    

    @register_mask("radial_2")
    @staticmethod
    def mask_radial(H, W, device):
        """Concentric: center/outer"""

        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        
        inner = (dist < 0.5).float().unsqueeze(0).unsqueeze(0)
        outer = (dist >= 0.5).float().unsqueeze(0).unsqueeze(0)
        
        return [inner, outer], ['center', 'outer']


    @register_mask("diagonal_2")
    @staticmethod
    def mask_diagonal(H, W, device):
        """Diagonal bands: upper-left/lower-right"""

        y = torch.linspace(0, 1, H, device=device).view(H, 1)
        x = torch.linspace(0, 1, W, device=device).view(1, W)

        val = y + x #(H, W) grid from 0 to 2

        m1 = (val < 1.0).float().unsqueeze(0).unsqueeze(0)
        m2 = (val >= 1).float().unsqueeze(0).unsqueeze(0)
        
        return [m1, m2], ['upper', 'lower'] 
    

    @register_mask("horizontal")
    @staticmethod
    def mask_horizontal(H, W, device):
        """3 horizontal bands: top/middle/bottom"""

        masks = []
        for i in range(3):
            m = torch.zeros(1, 1, H, W, device=device)
            m[:, :, int(H * i / 3):int(H * (i + 1) / 3), :] = 1.0
            masks.append(m)
        return masks, ['top', 'middle', 'bottom']


    @register_mask("vertical")
    @staticmethod
    def mask_vertical(H, W, device):
        """3 vertical bands: left/center/right"""

        masks = []
        for i in range(3):
            m = torch.zeros(1, 1, H, W, device=device)
            m[:, :, :, int(W * i / 3):int(W * (i + 1) / 3)] = 1.0
            masks.append(m)
        return masks, ['left', 'center', 'right']
    

    @register_mask("radial")
    @staticmethod
    def mask_radial(H, W, device):
        """Concentric: center/ring/outer"""

        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        
        inner = (dist < 0.45).float().unsqueeze(0).unsqueeze(0)
        ring = ((dist >= 0.45) & (dist < 0.85)).float().unsqueeze(0).unsqueeze(0)
        outer = (dist >= 0.85).float().unsqueeze(0).unsqueeze(0)
        
        return [inner, ring, outer], ['center', 'ring', 'outer']


    @register_mask("diagonal")
    @staticmethod
    def mask_diagonal(H, W, device):
        """Diagonal bands: upper-left/middle/lower-right"""

        y = torch.linspace(0, 1, H, device=device).view(H, 1)
        x = torch.linspace(0, 1, W, device=device).view(1, W)

        val = y + x #(H, W) grid from 0 to 2

        m1 = (val < 0.67).float().unsqueeze(0).unsqueeze(0)
        m2 = ((val >= 0.67) & (val <1.33)).float().unsqueeze(0).unsqueeze(0)
        m3 = (val >= 1.33).float().unsqueeze(0).unsqueeze(0)
        
        return [m1, m2, m3], ['upper', 'middle', 'lower']  


    @register_mask("quadrant")
    @staticmethod
    def mask_quadrant(H, W, device):
        """2x2 quadrants"""

        ymid, xmid = H // 2, W // 2
        
        tl = torch.zeros(1, 1, H, W, device=device)
        tl[:, :, :ymid, :xmid] = 1.0
        
        tr = torch.zeros(1, 1, H, W, device=device)
        tr[:, :, :ymid, xmid:] = 1.0
        
        bl = torch.zeros(1, 1, H, W, device=device)
        bl[:, :, ymid:, :xmid] = 1.0
        
        br = torch.zeros(1, 1, H, W, device=device)
        br[:, :, ymid:, xmid:] = 1.0
        
        return [tl, tr, bl, br], ['TL', 'TR', 'BL', 'BR']
    

    @register_mask("horizontal_4")
    @staticmethod
    def mask_horizontal_4(H, W, device):
        """4 horizontal bands"""

        masks = []
        for i in range(4):
            m = torch.zeros(1, 1, H, W, device=device)
            m[:, :, int(H * i / 4):int(H * (i + 1) / 4), :] = 1.0
            masks.append(m)
        return masks, ['band1', 'band2', 'band3', 'band4']


    @register_mask("vertical_4")
    @staticmethod
    def mask_vertical_4(H, W, device):
        """4 vertical bands"""

        masks = []
        for i in range(4):
            m = torch.zeros(1, 1, H, W, device=device)
            m[:, :, :, int(W * i / 4):int(W * (i + 1) / 4)] = 1.0
            masks.append(m)
        return masks, ['band1', 'band2', 'band3', 'band4']


    @register_mask("radial_4")
    @staticmethod
    def mask_radial_4(H, W, device):
        """4 concentric rings"""

        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        
        m1 = (dist < 0.35).float().unsqueeze(0).unsqueeze(0)
        m2 = ((dist >= 0.35) & (dist < 0.6)).float().unsqueeze(0).unsqueeze(0)
        m3 = ((dist >= 0.6) & (dist < 0.85)).float().unsqueeze(0).unsqueeze(0)
        m4 = (dist >= 0.85).float().unsqueeze(0).unsqueeze(0)
        
        return [m1, m2, m3, m4], ['core', 'inner', 'outer', 'edge']
    

    @staticmethod
    def get_semantic_masks(H, W, K, device, mask_type="afhq", regions=("eyes", "mouth")):
        """Get semantic masks for K classes"""
        
        if mask_type is None:
            return None, None 

        if mask_type == 'afhq':
            return MASK.get_afhq_semantic_masks(H, W, device, regions=regions)
        elif mask_type == "auto":
            if K == 3:
                mask_type = "horizontal"
            elif K == 4:
                mask_type = "quadrant"
            else:
                raise ValueError(f"No auto mask for K={K}.")
        
        if mask_type not in MASK.REGISTRY:
            available = list(MASK.REGISTRY.keys())
            raise ValueError(f"Unknown mask_type '{mask_type}'. Available: {available}")
        
        masks, names = MASK.REGISTRY[mask_type](H, W, device)
        
        if len(masks) != K:
            raise ValueError(f"Mask '{mask_type}' produces {len(masks)} regions but K={K} classes given.")
        
        # Verify partition
        total = sum(masks)
        if not torch.allclose(total, torch.ones_like(total), atol=1e-5):
            print(f"Warning: masks don't perfectly partition space, normalizing...")
            masks = [m / (total + 1e-8) for m in masks]
        
        return masks, names


    @staticmethod
    def list_available_masks():
        print(f"Available masks in registry: {list(MASK.REGISTRY.keys())}")


    @staticmethod
    def soften_masks_bruteforce(masks, sigma=1.25, eps=1e-8):
        """
        Repeat avg_pool blur then renormalize (sum==1). MASKBENDER.
        """

        if masks is None:
            return None
        
        if sigma <= 0:
            return masks

        k = max(3, int(2 * round(2 * sigma) + 1))
        reps = max(1, int(round(sigma)))

        soft = []
        for m in masks:
            x = m
            for _ in range(reps):
                x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
            soft.append(x)

        denom = torch.zeros_like(soft[0])
        for x in soft:
            denom = denom + x
        denom = denom.clamp_min(eps)

        return [x / denom for x in soft]


    @staticmethod
    def masks_viz_bw(masks, out_size, ignore=None, shitf=True):
        """B/W visualization for non-overlapping of masks."""
        if masks is None:
            return None
        
        dev = masks[0].device
        b_size = masks[0].shape[0]
        viz = torch.zeros(b_size, 1, out_size, out_size, device=dev)
        index = -1 if ignore else None
        
        for m in masks[:index]: # exclude the "rest" mask when asked
            up = F.interpolate(m, size=(out_size, out_size), mode="bilinear", align_corners=False)
            viz = viz + up

        viz = viz.clamp(0, 1).repeat(1, 3, 1, 1)
        return viz * 2.0 - 1 if shitf else viz

    @staticmethod
    def masks_viz_rgb(masks, out_size, shitf=True, seed=0):
        """RGB visualization of masks."""
        if masks is None:
            return None
        
        dev = masks[0].device
        K = len(masks)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        colors = (torch.rand(K, 3, generator=g) * 0.8 + 0.2).to(dev)
        b_size = masks[0].shape[0]
        viz = torch.zeros(b_size, 3, out_size, out_size, device=dev)
        for i, m in enumerate(masks):
            up = F.interpolate(m, size=(out_size, out_size), mode="bilinear", align_corners=False)
            viz = viz + up * colors[i].view(1, 3, 1, 1)
        
        viz = viz.clamp(0, 1)
        return viz * 2.0 - 1 if shitf else viz # shift from [0,1] to [-1, 1] to match image range