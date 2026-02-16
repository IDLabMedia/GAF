""" Around the latent in thousand classes. """

import torch
from transport.algebra import TA
from transport.solvers import SOLVER
from transport.mask import MASK
from utils.vae import decode


class CYCLE:
    REGISTRY = {}

    def register_ride(name, reg=REGISTRY):
        def decorator(fn):
            reg[name] = fn
            return fn
        return decorator


    @register_ride('cycle')
    @staticmethod
    @torch.no_grad()
    def cyclic_transport(cfg, gaf, vae, k_indices, steps=20, solver='euler', track=False):
        """
        Go round and round; round and round...
        """

        device = torch.device(cfg.device)
        gaf.eval()
        
        Hlat = Wlat = cfg.image_size // 8
        
        z0 = torch.randn(1, cfg.latent_ch, Hlat, Wlat, device=device, dtype=torch.float32)

        # start noise -> class 0.

        #check if the user has provided a cycle loop or not i.e, (0 1 2 0)?
        if k_indices[0]!=k_indices[-1]:
            k_indices.append(k_indices[0])

        transitions = list(zip(k_indices, k_indices[1:])) # cyclic transition pairs: (C1, C2), (C2, C3)... (Cn, C1)

        print("Transport map:", " -> ".join(map(str, k_indices)))
        lbl_first = torch.full((1,), k_indices[0], device=device, dtype=torch.long)
            
        def field_first(x, t):
            sv= getattr(field_first, 'solver_type', "velocity")
            if sv == "endpoint":
                J, K = gaf(x, t, y=lbl_first)
                return J, K

            return gaf.velocity(x, t, y=lbl_first) 

        z_current, trajectory = SOLVER.integrate(z0, steps, cfg.t_eps, device, field_first, solver=solver, track=track)
        
        all_imgs = [decode(vae, z_current, cfg)]
        latent_z = [z_current.clone()]

        for k_i, k_j in transitions:
            print(f"Cycle Transport: {k_i} -> {k_j}")

            lbl_i = torch.full((1,), k_i, device=device, dtype=torch.long)
            lbl_j = torch.full((1,), k_j, device=device, dtype=torch.long)
                    
            def field_rev(x, t):
                sv= getattr(field_rev, 'solver_type', "velocity")
                if sv == "endpoint":
                    J, K = gaf(x, t, y=lbl_i)
                    return J, K

                return gaf.velocity(x, t, y=lbl_i) 
                           
            def field_fwd(x, t):
                sv= getattr(field_fwd, 'solver_type', "velocity")
                if sv == "endpoint":
                    J, K = gaf(x, t, y=lbl_j)
                    return J, K

                return gaf.velocity(x, t, y=lbl_j) 
            
            z_noise, reverse_trajectory = SOLVER.integrate(z_current, steps, cfg.t_eps, device, field_rev, solver=solver, direction="reverse", track=track)
            trajectory.extend(reverse_trajectory)

            z_current, forward_trajectory = SOLVER.integrate(z_noise, steps, cfg.t_eps, device, field_fwd, solver=solver, track=track)
            trajectory.extend(forward_trajectory)

            latent_z.append(z_current.clone())

            with torch.no_grad():
                img = decode(vae, z_current, cfg)
                all_imgs.append(img)
            

        main_result = {
            'img': torch.cat(all_imgs, dim=0),
             'z': torch.cat(latent_z, dim=0)
        }

        return main_result, trajectory
    

    @register_ride('interpolation')
    @torch.no_grad()
    def interpolation_cyclic_transport(cfg, gaf, vae, k_indices, steps=100, method='euler', n_interp=10, track=False):
        """
        Chain through J while interpolating. 
        """

        device = cfg.device
        Hlat = Wlat = cfg.image_size // 8
        z_noise = torch.randn(1, cfg.latent_ch, Hlat, Wlat, device=device, dtype=torch.float32)
        
        all_imgs = []
        all_latents = []


        #check if the user has provided a cycle loop or not i.e, (0 1 2 0)?
        if k_indices[0]!=k_indices[-1]:
            k_indices.append(k_indices[0])

        transitions = list(zip(k_indices, k_indices[1:])) # cyclic transition pairs: (C1, C2), (C2, C3)... (Cn, C1)

        print("Transport map:", " -> ".join(map(str, k_indices)))
        trajectory = []
        for k_i, k_j in transitions:
            print(f"Interpolation Transport: [{k_i} -> {k_j}]")

            result, interp_trajectory = CYCLE.k_head_interpolation(cfg, gaf, vae, k_i, k_j, z_noise, steps, method, n_interp, track=track)
            trajectory.extend(interp_trajectory)
            all_imgs.append(result['img'])
            all_latents.append(result['z'])
            z_noise = result['z_noise']  # Chain!
        
        main_result = {
            'img': torch.cat(all_imgs, dim=0),
             'z': torch.cat(all_latents, dim=0) 
        }

        return main_result, trajectory


    @register_ride('barycentric')
    @torch.no_grad()             
    def barycentric_interpolation(cfg, gaf, vae, k_indices, steps=100, solver='euler', grid_size=7, track=False):
        """
        3-way barycentric interpolation between three K heads over a full grid.
        Square -> simplex mapping: alpha=u(1-v), beta=(1-u)(1-v), gamma=v.
        """

        device = cfg.device
        Hlat = Wlat = cfg.image_size // 8
        z_noise = torch.randn(1, cfg.latent_ch, Hlat, Wlat, device=device, dtype=torch.float32)

        gaf.eval()

        k0, k1, k2 = k_indices

        imgs = []
        trajectory = []
        l0 = torch.full((1,), k0, device=device, dtype=torch.long)
        l1 = torch.full((1,), k1, device=device, dtype=torch.long)
        l2 = torch.full((1,), k2, device=device, dtype=torch.long)
        
        for i in range(grid_size):
            u = i / (grid_size - 1)
            for j in range(grid_size):
                v = j / (grid_size - 1)
                alpha = u * (1.0 - v)
                beta  = (1.0 - u) * (1.0 - v)
                gamma = v

                s = alpha + beta + gamma
                a, b, g = alpha/s, beta/s, gamma/s

                def barycentric_drift(x, t, a=a, b=b, g=g):
                    v0 = gaf.velocity(x, t, y=l0)
                    v1 = gaf.velocity(x, t, y=l1)
                    v2 = gaf.velocity(x, t, y=l2)
                    return a * v0 + b * v1 + g * v2
                
                z, bary_trajectory = SOLVER.integrate(z_noise.clone(), steps, cfg.t_eps, device, barycentric_drift, solver)
                trajectory.extend(bary_trajectory)
                img = decode(vae, z, cfg)
                imgs.append(img)

        main_result = {
            'img': torch.cat(imgs, dim=0),
             'z': torch.tensor([]) # keeping up with the errors.
        }

        return main_result, trajectory


    @staticmethod
    @torch.no_grad()
    def k_head_interpolation(cfg, gaf, vae, k_i, k_j, z_noise, steps=20, solver='euler', n_interp=10, track=False):
        """
        Chain through J, aka the Universal passport. 
        """

        device = cfg.device
        gaf.eval()
        
        alphas = torch.linspace(0, 1, n_interp, device=device)
        all_imgs = []
        latent_z = []
        
        lbl_i = torch.full((1,), k_i, device=device, dtype=torch.long)
        lbl_j = torch.full((1,), k_j, device=device, dtype=torch.long)
        
        def get_drift(alpha):
            def drift(x, t):
                v_i = gaf.velocity(x, t, y=lbl_i)
                v_j = gaf.velocity(x, t, y=lbl_j)
                return (1 - alpha) * v_i + alpha * v_j
            return drift
        
        z_J = z_noise
        z_current = None
        prev_alpha = None
        
        for alpha in alphas:
            a = alpha.item()
            
            if prev_alpha is None:
                # First: noise -> image
                drift_fwd = get_drift(a)
                z_current, trajectory = SOLVER.integrate(z_J, steps, cfg.t_eps, device, drift_fwd, solver=solver, track=track)
            else:
                # Reverse prev -> J -> forward new
                drift_rev = get_drift(prev_alpha)
                z_J, reverse_trajectory = SOLVER.integrate(z_current, steps, cfg.t_eps, device, drift_rev, solver=solver, direction="reverse", track=track)
                trajectory.extend(reverse_trajectory)
                drift_fwd = get_drift(a)
                z_current, forward_trajectory = SOLVER.integrate(z_J, steps, cfg.t_eps, device, drift_fwd, solver=solver, track=track)
                trajectory.extend(forward_trajectory)

            latent_z.append(z_current.clone())
            all_imgs.append(decode(vae, z_current, cfg))
            prev_alpha = a
        
        # Recover J for next transition (last alpha = 1.0 = pure K_j)
        drift_last = get_drift(prev_alpha)
        z_J_final, final_leg = SOLVER.integrate(z_current, steps, cfg.t_eps, device, drift_last, solver=solver, direction="reverse", track=track)
        trajectory.extend(final_leg)
        
        main_result = {
            'img': torch.cat(all_imgs, dim=0),
            'z': torch.cat(latent_z, dim=0),
            'z_noise': z_J_final  # Pass to next transition.
        }

        return main_result, trajectory


    @classmethod
    def ride(cycle, mode, *args, **kwargs):
        if mode not in cycle.REGISTRY:
            raise KeyError(f"Unknown ride: {mode}")
        
        return cycle.REGISTRY[mode](*args, **kwargs)
    