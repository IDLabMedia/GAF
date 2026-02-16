import torch

def sample_t_uniform(cfg, batch_size, device):
    t_eps = float(cfg.t_eps)
    t = torch.rand(batch_size, device=device, dtype=torch.float32)
    return t.mul_(1 - 2*t_eps).add_(t_eps)

def init_loss_dict():
    return {"pair": 0, "noise_loss": 0, "data_loss": 0, "lambda_res": 0, "lambda_time": 0,
            "residual_pen": 0, "time_antisym": 0, "v_mse": 0, "cos": 0
            }

def gaf_loss(model, z_x, z_y, t, y, cfg, curr_step):
    """
    Implements:
      L_pair = (1-t)||J - z_y||_2^2 + t||K - z_x||_2^2
      L_res  = (1-t)||J_res||_2^2 + t||K_res||_2^2
      L_swap = ||J_res + K_res_sw||_2^2 + ||K_res + J_res_sw||_2^2
    """  
    logs = init_loss_dict()

    t4  = t[:, None, None, None]
    z_t = (1 - t4) * z_y + t4 * z_x

    J, K = model(z_t, t, y)

    wJ, wK = (1 - t4), t4

    # pair loss
    noise_loss = (wJ * (J - z_y).pow(2)).float().mean()
    data_loss  = (wK * (K - z_x).pow(2)).float().mean()

    L = noise_loss + data_loss

    logs["pair"] = float((noise_loss + data_loss).detach())
    logs["noise_loss"] =  float(noise_loss.detach())
    logs["data_loss"] = float(data_loss.detach())

    # CHECK IF VELOCITY IS WHAT WE ARE CLAIMING TO BE, I.E. v=K-J ~ z_x-z_y (best when cos >0.95)
    with torch.no_grad():
        mse_v, cos_v = check_velocity(K, J, z_x, z_y)
        logs["v_mse"] = float(mse_v.detach())
        logs["cos"] = float(cos_v.detach())
    # END CHECK :)

    J_res = J - (1.0 - t4) * z_t
    K_res = K - t4 * z_t
    
    if cfg.res_reg > 0: #(set res_reg zero for residual penality ablation.) 
        # res penality
        edge_pen = (wJ * J_res.pow(2) + wK * K_res.pow(2)).float().mean()
        L = L + cfg.res_reg * edge_pen

        logs["lambda_res"] = cfg.res_reg
        logs["residual_pen"] = float((cfg.res_reg * edge_pen).detach())

    if cfg.lam_time > 0: #(set swap zero for swap loss ablation.) 
        # swap loss
        J_sw, K_sw = model(z_t, (1.0 - t), y)

        J_res_sw = J_sw - (t4 * z_t)
        K_res_sw = K_sw - ((1.0 - t4) * z_t)

        L_time = ((J_res + K_res_sw).pow(2) + (K_res + J_res_sw).pow(2)).float().mean()
        L = L + cfg.lam_time * L_time

        logs["lambda_time"] = cfg.lam_time
        logs["time_antisym"] = float((cfg.lam_time * L_time).detach())

    return L, logs


def check_velocity(K, J, z_x, z_y):
    v_true = (z_x - z_y).float()
    v_hat  = (K - J).float()

    mse_v = (v_true - v_hat).pow(2).mean()

    cos_v = torch.nn.functional.cosine_similarity(
        v_true.reshape(v_true.size(0), -1),
        v_hat.reshape(v_hat.size(0), -1),
        dim=1
    ).mean()
    
    return mse_v, cos_v

