import torch

class RBM(torch.nn.Module):
    """
    Restricted Boltzmann Machine (RBM) for modeling the quantum wave function.
    """
    def __init__(self, n_visible, n_hidden, device='cpu'):
        super(RBM, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(n_visible, dtype=torch.complex128, device=device) * 0.05)
        self.b = torch.nn.Parameter(torch.randn(n_hidden, dtype=torch.complex128, device=device) * 0.05)
        self.W = torch.nn.Parameter(torch.randn(n_visible, n_hidden, dtype=torch.complex128, device=device) * 0.05)
        self.to(device)

    def log_prob(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculates the logarithm of the unnormalized probability (log-amplitude) 
        for a batch of spin configurations.
        
        Args:
            sigma: A tensor of shape [n_batch, n_visible]
            
        Returns:
            A tensor of shape [n_batch] containing the log-amplitude for each configuration.
        """
        sigma_complex = sigma.to(dtype=torch.complex128)
        
        # visible_term shape: [n_batch]
        visible_term = torch.einsum('bi,i->b', sigma_complex, self.a)
        
        # theta shape: [n_batch, n_hidden]
        theta = torch.einsum('bi,ij->bj', sigma_complex, self.W) + self.b
        
        # hidden_term shape: [n_batch]
        hidden_term = torch.sum(torch.log(2 * torch.cosh(theta)), dim=1)
        
        return visible_term + hidden_term