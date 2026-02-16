import copy
import torch

class EMA:
    def __init__(self, model, decay_warmup, decay_main, warmup_steps):
    
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters(): p.requires_grad_(False)
        
        self.decay_warmup = decay_warmup
        self.decay_main = decay_main
        self.warmup_steps = warmup_steps

    @torch.no_grad()
    def update(self, model, step):
        decay = self.decay_main if step >= self.warmup_steps else self.decay_warmup
        
        msd = model.state_dict()
        esd = self.ema_model.state_dict()

        for key in msd:
            p_src = msd[key]
            p_ema = esd[key]

            if p_src.dtype.is_floating_point:
                p_ema.lerp_(p_src, 1.0 - decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict, **kwargs):
        self.ema_model.load_state_dict(state_dict, **kwargs)
    
    def eval(self):
        self.ema_model.eval()
        return self

    def train(self):
        self.ema_model.train()
        return self
    
    def __getattr__(self, name):
        return getattr(self.ema_model, name)