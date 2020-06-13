import torch
import torch.nn as nn

class ReluMLP(nn.Module):
    """
    Initialize a multilayer perceptron with ReLU activation function."""

    def __init__(self, n_inputs, n_hidden_nodes, n_outputs, bias=True):
        super(ReluMLP, self).__init__()

        self.layer1 = nn.Linear(n_inputs, n_hidden_nodes, bias)
        self.layer2 = nn.Linear(n_hidden_nodes, n_hidden_nodes, bias)
        self.layer3 = nn.Linear(n_hidden_nodes, n_outputs, bias)
    
    def forward(self, X):
        X = X.view(X.size(0), -1)
        X = torch.relu(self.layer1(X))
        X = torch.relu(self.layer2(X))
        X = self.layer3(X)
        return X

class SigmoidMLP(nn.Module):
    """
    Initialize a multilayer perceptron with sigmoid activation function."""

    def __init__(self, n_inputs, n_hidden_nodes, n_outputs, bias=True):
        super(SigmoidMLP, self).__init__()

        self.layer1 = nn.Linear(n_inputs, n_hidden_nodes, bias)
        self.layer2 = nn.Linear(n_hidden_nodes, n_hidden_nodes, bias)
        self.layer3 = nn.Linear(n_hidden_nodes, n_outputs, bias)
    
    def forward(self, X):
        X = X.view(X.size(0), -1)
        X = torch.sigmoid(self.layer1(X))
        X = torch.sigmoid(self.layer2(X))
        X = self.layer3(X)
        return X

class TanhMLP(nn.Module):
    """
    Initialize a multilayer perceptron with tanh activation function."""

    def __init__(self, n_inputs, n_hidden_nodes, n_outputs, bias=True):
        super(TanhMLP, self).__init__()

        self.layer1 = nn.Linear(n_inputs, n_hidden_nodes, bias)
        self.layer2 = nn.Linear(n_hidden_nodes, n_hidden_nodes, bias)
        self.layer3 = nn.Linear(n_hidden_nodes, n_outputs, bias)
    
    def forward(self, X):
        X = X.view(X.size(0), -1)
        X = torch.tanh(self.layer1(X))
        X = torch.tanh(self.layer2(X))
        X = self.layer3(X)
        return X