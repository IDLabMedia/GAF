import torch
import torch.nn as nn


class J(nn.Module):
    """
    The Noise twin (Anchor J)
    Shared across all classes and initialized to zero for boundary stability at t=0.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=False)
        nn.init.constant_(self.linear.weight, 0)

    def forward(self, x): return self.linear(x)


class K(nn.Module):
    """
    The Data twin (Anchor K)
    Targets the data manifold at t=1. We want clear separation of K from J at iter 1 (v=K-J), so no init zeroing K.
    Vectorized to bypass for loop usage for num_class>1.
    """

    def __init__(self, hidden_size, patch_size, out_channels, num_classes):
        super().__init__()

        self.num_classes = num_classes
        out_dim = patch_size * patch_size * out_channels

        self.weight = nn.Parameter(torch.empty(num_classes, hidden_size, out_dim))
        self.bias = nn.Parameter(torch.empty(num_classes, out_dim))

        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def __len__(self): return self.num_classes
    
    def forward(self, x, y):
        w = self.weight[y]
        b = self.bias[y].unsqueeze(1)
        return torch.matmul(x, w) + b


class Twins(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, num_classes=1):
        super().__init__()
        self.J = J(hidden_size, patch_size, out_channels)
        self.K = K(hidden_size, patch_size, out_channels, num_classes)

    def forward(self, f_t, y):
        return self.J(f_t), self.K(f_t, y)