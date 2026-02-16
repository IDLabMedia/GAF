
import torch
import torch.nn as nn
from .trunk import DiTRUNK
from .twins import Twins

class GAF(nn.Module):
    def __init__(self, latent_ch, input_size, hidden_size, depth, num_heads, patch_size = 2, mlp_ratio = 4.0, num_classes = 1000, learn_sigma = True):
        super().__init__()
        self.trunk = DiTRUNK(input_size=input_size, patch_size=patch_size, in_channels=latent_ch,
                             hidden_size=hidden_size, depth=depth, num_heads=num_heads,
                             mlp_ratio=mlp_ratio, num_classes=num_classes,
                             learn_sigma=learn_sigma)
        
        self.twins = Twins(hidden_size, patch_size, latent_ch, num_classes)

        nn.init.constant_(self.trunk.ada_ln.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.trunk.ada_ln.adaLN_modulation[-1].bias,  0)

    def forward(self, x_t, t, y):
        f_t = self.trunk(x_t, t, y)
        J_tok, K_tok = self.twins(f_t, y)
        
        J_res = self.trunk.unpatchify(J_tok)
        K_res = self.trunk.unpatchify(K_tok)

        t = t.view(-1, 1, 1, 1)
        J = (1.0 - t) * x_t + J_res
        K = t * x_t + K_res

        return J, K
    
    @torch.no_grad()
    def velocity(self, x_t, t, y):
        J, K = self.forward(x_t, t, y)

        return K - J