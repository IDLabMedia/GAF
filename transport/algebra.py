"""
Transport Algebra
"""

import torch
from transport.solvers import SOLVER
from utils.vae import decode

class TA:


    @staticmethod
    @torch.no_grad()
    def generate_pure(z_noise, cfg, gaf, vae, classes, steps, method="euler"):
        """ Generate specific class 
            ENDPOINT use K via IER
            VELOCITY use v=K-J
        """

        k = classes
        lbl = torch.full((z_noise.shape[0],), k, device=z_noise.device, dtype=torch.long)

        def field(x, t):
            sv= getattr(field, 'solver_type', "velocity")

            if sv == "endpoint":
                return gaf(x, t, y=lbl) # J, K
            else:
                return gaf.velocity(x, t, y=lbl)
                
        z, trajectory = SOLVER.integrate(z_noise.clone(), steps, cfg.t_eps, z_noise.device, field, method)
        
        return z, trajectory


    @staticmethod
    @torch.no_grad()
    def generate_scalar_blend(z_noise, cfg, gaf, vae, classes, alpha, steps, method="euler"):
        """Scalar  VELOCITY OR ENDPOINT  interpolation: v = a*v1 + (1-a)*v2
           ENDPOINT use K via IER
           VELOCITY use v=K-J
        """

        device = z_noise.device
        k1, k2 = classes
        B = z_noise.shape[0]
        lbl1 = torch.full((B,), k1, device=device, dtype=torch.long)
        lbl2 = torch.full((B,), k2, device=device, dtype=torch.long)
        
        def field(x, t):
            sv= getattr(field, 'solver_type', "velocity")

            if sv == "endpoint":
                J1, K1 = gaf(x, t, y=lbl1)
                J2, K2 = gaf(x, t, y=lbl2)
                return alpha * J1 + (1.0 - alpha) * J2, alpha * K1 + (1.0 - alpha) * K2
            else:
                v1 = gaf.velocity(x, t, y=lbl1)
                v2 = gaf.velocity(x, t, y=lbl2)
                return alpha * v1 + (1.0 - alpha) * v2
        
        z, trajectory = SOLVER.integrate(z_noise.clone(), steps, cfg.t_eps, device, field, method)
        
        return z, trajectory


    @staticmethod
    @torch.no_grad()
    def generate_spatial(z_noise, cfg, gaf, vae, classes, masks, steps, method="euler"):
        """Spatial VELOCITY OR ENDPOINT composition with masks. Works for any K>=2 classes.
           ENDPOINT use K via IER
           VELOCITY use v=K-J
        """

        device = z_noise.device
        B = z_noise.shape[0]
        K = len(classes)
        labels = [torch.full((B,), c, device=device, dtype=torch.long) for c in classes]
        
        def field(x, t):
            sv= getattr(field, 'solver_type', "velocity")

            if sv == "endpoint":
                Kblend = torch.zeros_like(x)
                for i in range(K):
                    J, Ki = gaf(x, t, y=labels[i]) # J is not used
                    Kblend = Kblend + masks[i] * Ki
                return J, Kblend
            
            v = torch.zeros_like(x)
            for i in range(K):
                v = v + masks[i] * gaf.velocity(x, t, y=labels[i])
            return v
        
        z, trajectory = SOLVER.integrate(z_noise.clone(), steps, cfg.t_eps, device, field, method)
        
        return z, trajectory


    @staticmethod
    @torch.no_grad()
    def generate_weighted_blend(z_noise, cfg, gaf, vae, classes, weights, steps, method="euler", normalize=True): #(cfg, gaf, vae, base_k, attributes, steps=100):
        """
        Compose multiple attributes with independent control.
        attributes: list of (k_idx, weight)
        Example: 70% cat + 30% dog + 20% wild
           ENDPOINT use K via IER
           VELOCITY use v=K-J
        """
        
        if normalize:
            total = sum(weights)
            weights = [w/total for w in weights]

        if len(weights) < len(classes): # redistribute the weights.
            remainder = 1.0 - sum(weights) 
            n_missing = len(classes) - len(weights)
            fill = remainder / n_missing
            weights = list(weights) + [fill] * n_missing
            print(f"Weights: {weights[:len(weights)-n_missing]} + {n_missing}*{fill:.3f}(remainder).")

        device = z_noise.device
        B = z_noise.shape[0]
        labels = [torch.full((B,), c, device=device, dtype=torch.long) for c in classes]
        
        def field(x, t):
            sv= getattr(field, 'solver_type', "velocity")

            if sv == "endpoint":
                blend_term = [f"{w}*K{c}" for c,w in zip(classes, weights)] # display current weighted K blend eqn.
                print("Kblend = " + " + ".join(blend_term),"\n")

                Kblend = torch.zeros_like(x)
                for lbl, w in zip(labels, weights):
                    J, K = gaf(x, t, y=lbl) # J is not used
                    Kblend = Kblend + w * K
                return J, Kblend

            v_term = [f"{w}*v{c}" for c,w in zip(classes, weights)] # display current weighted velocity eqn.
            print("v = " + " + ".join(v_term),"\n")

            v = torch.zeros_like(x)
            for lbl, w in zip(labels, weights):
                v = v + w * gaf.velocity(x, t, y=lbl)
            return v
            
        z, trajectory = SOLVER.integrate(z_noise.clone(), steps, cfg.t_eps, device, field, method)
        
        return z, trajectory
    

    @staticmethod
    @torch.no_grad()
    def generate_spacial_weighted(z_noise, cfg, gaf, vae, classes, masks, weights, steps, method="euler", normalize=True): #(cfg, gaf, vae, base_k, attributes, steps=100):
        """
        Compose multiple attributes with user provided mask.
        attributes: list of (k_idx, mask, weight)
        Example: 70% * cat_mask * cat + 30% * dog_mask * dog + 20% * wild_mask * wild
        Mask controls WHERE each class appears
        Weights control HOW MUCH each class contributes
           ENDPOINT use K via IER
           VELOCITY use v=K-J
        """
        
        device = z_noise.device
        B = z_noise.shape[0]
        labels = [torch.full((B,), c, device=device, dtype=torch.long) for c in classes]
        
        def field(x, t):
            sv= getattr(field, 'solver_type', "velocity")

            if sv == "endpoint":
                Kblend = torch.zeros_like(x)
                for lbl, mask, w in zip(labels, masks, weights):
                    J, K = gaf(x, t, y=lbl) # J is not used
                    Kblend += Kblend + w * mask * K
                return J, Kblend

            v = torch.zeros_like(x)
            for lbl, mask, w in zip(labels, masks, weights):
                v = v + w * mask * gaf.velocity(x, t, y=lbl)
            return v
        
        z, trajectory = SOLVER.integrate(z_noise.clone(), steps, cfg.t_eps, device, field, method)
    
        return z, trajectory

    
    @staticmethod
    @torch.no_grad() 
    def translate(z_source, cfg, gaf, vae, source_class, target_class, steps, method="euler"):
        """
        image-to-image translation.
        
        Invert source image to noise using source_class NOISE endpoint J OR the velocity (backward)
        Decode from noise to target using target_class DATA endpoint K OR the velocity (forward)
        """

        device = z_source.device
        
        lbl_src = torch.full((z_source.shape[0],), source_class, device=device, dtype=torch.long)

        def field_src(x, t):
            sv= getattr(field_src, 'solver_type', "velocity")

            if sv == "endpoint":
                J, K = gaf(x, t, y=lbl_src)
                return J, K
            
            # Source velocity (for inversion)
            #v_src = lambda x, t: gaf.velocity(x, t, y=lbl_src)
            return gaf.velocity(x, t, y=lbl_src)
    
        lbl_tgt = torch.full((z_source.shape[0],), target_class, device=device, dtype=torch.long)
        def field_tgt(x, t):
            sv= getattr(field_tgt, 'solver_type', "velocity")

            if sv == "endpoint":
                J, K = gaf(x, t, y=lbl_tgt)
                return J, K
            
            # Target velocity (for generation)
            #v_tgt = lambda x, t: gaf.velocity(x, t, y=lbl_tgt)
            return gaf.velocity(x, t, y=lbl_tgt)
        
        # Invert source to noise (t: 1 -> 0)
        # Negative velocity = backward integration
        z_noise, trajectory_J = SOLVER.integrate(z_source.clone(), steps, cfg.t_eps, device, field_src,  solver=method, direction="reverse")
        
        # Decode to target class (t: 0 -> 1)
        z_target, trajectory_K = SOLVER.integrate(z_noise, steps, cfg.t_eps, device, field_tgt, method)
        
        tensor_J = torch.cat(trajectory_J, dim=0)
        tensor_K = torch.cat(trajectory_K, dim=0)
        traj_grid = torch.cat([tensor_J, tensor_K], dim=0)

        return decode(vae, z_target, cfg), decode(vae, traj_grid, cfg)