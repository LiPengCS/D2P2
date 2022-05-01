import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

class FiveLayerNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=(100, 100, 100, 100)):
        super(FiveLayerNet, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.linear4 = nn.Linear(hidden_dim[2], hidden_dim[3])
        self.linear5 = nn.Linear(hidden_dim[3], output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        out = self.relu(out)
        out = self.linear5(out)
        return out

class EquiNNNorm(nn.Module):
    def __init__(self, input_dim, prep_space, end_model):
        super(EquiNNNorm, self).__init__()
        self.tfs = [nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh(), nn.Identity()]
        equi_layers = []

        for step in prep_space:
            params = nn.Parameter(torch.randn(1, input_dim, len(self.tfs)), requires_grad=True)
            equi_layers.append(params)

        self.first_layer = nn.Parameter(torch.randn(1, input_dim), requires_grad=True)
        self.equi_layers = nn.ParameterList(equi_layers)
        self.end_model = end_model

    def forward(self, x):
        out = x * self.first_layer

        for w in self.equi_layers:
            out = torch.cat([f(out).unsqueeze(-1) for f in self.tfs], dim=-1)
            out.retain_grad()
            out = torch.sum(out * w, dim=-1)

        out = self.end_model(out)
        return out