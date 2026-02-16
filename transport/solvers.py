"""
Numerical Solvers for ODE integration (Euler, Huen, RK4)
"""

import torch
from tqdm import tqdm

class SOLVER:

    REGISTRY = {}

    def register_solver(name, reg=REGISTRY):
        def decorator(fn):
            reg[name] = fn
            return fn
        return decorator


    # move forward or in reverse
    @staticmethod
    def time_grid_direction(steps, t_eps, device, direction="forward"):

        # Define Time Grid (0->1 or 1->0)
        if direction == "forward":
            ts = torch.linspace(t_eps, 1-t_eps, steps + 1, device=device, dtype=torch.float32)
        else:
            ts = torch.linspace(1-t_eps, t_eps, steps + 1, device=device, dtype=torch.float32) # reverse, I'm going going back back to J J
            
        return ts

    def tracker(solver, iterator, direction,  track=False, total=None):

        if track:
            iterator = tqdm(iterator, desc=f'{solver} {direction}', total=total, leave=False)
        return iterator


    @register_solver("endpoint")
    @staticmethod
    @torch.no_grad()
    def endpoint(z0, steps, t_eps, device, jk_fn, direction="forward", track=False, alpha=0.95):
        """
        Non-ODE generation via endpoint solver (uses K or J only).
        """
        trajectory = [z0]
        B = z0.size(0)
        
        if direction == "forward":
            x1 = z0.clone()                
            t_list = torch.linspace(t_eps, 1-t_eps, steps, device=device)
            it = SOLVER.tracker('Endpoint', t_list, direction, track, total=len(t_list))
            
            for t in it:
                t_scalar = float(t.item())
                tvec = torch.full((B,), t_scalar, device=device, dtype=torch.float32)
                t4 = tvec.view(-1,1,1,1).to(z0.dtype)
                
                xt = (1.0 - t4) * z0 + t4 * x1
                _, K = jk_fn(xt, tvec)
                
                x1 = (1-alpha) * x1 + alpha * K.to(x1.dtype)
                
                trajectory.append(x1.clone())

            return x1, trajectory
        
        else: #reverse (refine for noise)
            x_anchor = z0.clone() 
            j_est = torch.randn_like(x_anchor)                 
            t_list = torch.linspace(1-t_eps, t_eps, steps, device=device) # reverse
            it = SOLVER.tracker('Endpoint', t_list, direction, track, total=len(t_list))
            
            for t in it:
                t_scalar = float(t.item())
                tvec = torch.full((B,), t_scalar, device=device, dtype=torch.float32)
                t4 = tvec.view(-1,1,1,1).to(x_anchor.dtype)
                
                xt = (1.0 - t4) * j_est + t4 * x_anchor
                J, _ = jk_fn(xt, tvec)
                
                j_est = (1-alpha) * j_est + alpha * J.to(x_anchor.dtype)
                
                trajectory.append(j_est.clone())

            return j_est, trajectory


    @register_solver("euler")
    @staticmethod
    @torch.no_grad()
    def euler(z_y, steps, t_eps, device, v_fn, direction="forward", track=False):
        """
        Euler integration from t_eps to 1-t_eps..
        """

        #trajectory = [z_y.clone()]
        trajectory = []

        ts = SOLVER.time_grid_direction(steps, t_eps, device, direction)
        iter = SOLVER.tracker('Euler', range(steps), direction, track, total=steps)

        for k in iter:
            t0 = ts[k].expand(z_y.size(0))
            h  = float(ts[k+1] - ts[k]) # Time delta (positive or negative)
            v0 = v_fn(z_y,  t0).to(torch.float32)
            z_y = z_y + h * v0

            trajectory.append(z_y.clone()) # maybe save every step?

        return z_y, trajectory


    @register_solver("heun")
    @staticmethod
    @torch.no_grad()
    def huen(z_y, steps, t_eps, device, v_fn, direction="forward", track=False):
        """
        Huen integration from t_eps to 1-t_eps..
        """

        trajectory = [z_y.clone()]

        ts = SOLVER.time_grid_direction(steps, t_eps, device, direction)
        iter = SOLVER.tracker('Heun', range(steps), direction, track, total=steps)

        for k in iter:
            t0 = ts[k].expand(z_y.size(0))
            t1 = ts[k+1].expand(z_y.size(0))
            h  = float(ts[k+1] - ts[k])
            v0 = v_fn(z_y,  t0).to(torch.float32)
            z_e = z_y + h * v0
            v1 = v_fn(z_e, t1).to(torch.float32)
            z_y  = z_y + 0.5 * h * (v0 + v1)

            trajectory.append(z_y.clone())

        return z_y, trajectory


    @register_solver("rk4")
    @staticmethod
    @torch.no_grad()
    def rk4(z_y, steps, t_eps, device, v_fn, direction="forward", track=False):
        """
        4th-Order Runge-Kutta integration from t_eps to 1-t_eps..
        """

        trajectory = [z_y.clone()]
        
        ts = SOLVER.time_grid_direction(steps, t_eps, device, direction)
        iter = SOLVER.tracker('RK4', range(steps), direction, track, total=steps)

        for k in iter:
            t_curr = ts[k].expand(z_y.size(0))
            h = float(ts[k + 1] - ts[k])
            
            k1 = v_fn(z_y, t_curr).to(torch.float32)
            t_half = t_curr + 0.5 * h
            k2 = v_fn(z_y + 0.5 * h * k1, t_half).to(torch.float32) 
            k3 = v_fn(z_y + 0.5 * h * k2, t_half).to(torch.float32) 
            t_next = ts[k + 1].expand(z_y.size(0))
            k4 = v_fn(z_y + h * k3, t_next).to(torch.float32) 
            z_y = z_y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            trajectory.append(z_y.clone())
            
        return z_y, trajectory


    @staticmethod
    def integrate(z_y, steps, t_eps, device, v_fn, solver="euler", direction="forward", track=False):
        """
        Main entry point for the SOLVERs.
        """
        
        if solver not in SOLVER.REGISTRY:
            raise ValueError(f"Solver {solver} not found. Available solvers: {list(SOLVER.REGISTRY.keys())}")
        
        JK_SOLVERS = {'endpoint'} # non velocity solver, using J and K
        
        if solver in JK_SOLVERS:
            setattr(v_fn, "solver_type", "endpoint")
        else:
            setattr(v_fn, "solver_type", "velocity")
        
        z_y = z_y.to(torch.float32)
        return SOLVER.REGISTRY[solver](z_y, steps, t_eps, device, v_fn, direction, track)