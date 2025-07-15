import torch

class RBM(torch.nn.Module):
    """
    Restricted Boltzmann Machine (RBM) for modeling the quantum wave function.
    """
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(n_visible, dtype=torch.cfloat) * 0.05)
        self.b = torch.nn.Parameter(torch.randn(n_hidden, dtype=torch.cfloat) * 0.05)
        self.W = torch.nn.Parameter(torch.randn(n_visible, n_hidden, dtype=torch.cfloat) * 0.05)

    def log_prob(self, sigma):
        """
        Calculates the logarithm of the unnormalized probability (log-amplitude) of a given spin configuration.
        log(Psi(sigma))
        """
        visible_term = torch.dot(sigma.cfloat(), self.a)
        theta = torch.einsum('i,ij->j', sigma.cfloat(), self.W) + self.b
        hidden_term = torch.sum(torch.log(2 * torch.cosh(theta)))
        return visible_term + hidden_term