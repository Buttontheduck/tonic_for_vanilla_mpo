import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Temperature(torch.nn.Module):
    def __init__(self, encoder,hidden_dim, n_hidden ,device):
        super().__init__()
        self.encoder = encoder
        self.n_hidden = n_hidden       
        self.hidden_dim = hidden_dim
        self.device = device

    def initialize(
        self, observation_space):
        size = self.encoder.initialize(
            observation_space)
        self.model = TempNet(in_dim=size,out_dim=1,hidden_dim=self.hidden_dim,n_hidden=self.n_hidden).to(self.device)
        
    def forward(self, *inputs):
        out = self.model(*inputs)
        return out
    
    
class TempNet(nn.Module):
    def __init__(self,in_dim, out_dim, hidden_dim=256, n_hidden=2):
        super().__init__()

        self.eta_eps = 1e-8
        # Main network with additional input for condition
        layers = []

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Mish())
        
        # Hidden layers
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Mish())
        
        # Output layer
        last_linear = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
        layers.append(last_linear)
        
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):

        η_free = self.network(x)
        
        η = F.softplus(η_free) + self.eta_eps    # positive: η ∈ (ε, ∞)

        return η
        

    
