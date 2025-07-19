import torch
from tqdm import trange

def parallel_metropolis_hastings_sampler(model, n_samples_per_chain, n_chains, n_spins, burn_in_steps, step_interval, device='cpu'):
    """
    Generates spin configurations using n_chains parallel Metropolis-Hastings samplers.
    """
    # Initialize all chains with random spin configurations in {-1, +1}
    # Shape: [n_chains, n_spins]
    current_configs = torch.randint(0, 2, (n_chains, n_spins), device=device).float() * 2 - 1
    
    with torch.no_grad():
        log_probs_current = model.log_prob(current_configs).real

        # Burn-in phase
        for _ in range(burn_in_steps):
            # Propose a single spin flip for each chain
            flip_indices = torch.randint(0, n_spins, (n_chains,), device=device)
            
            # Use scatter_ to flip the spins in-place for efficiency
            updates = -2 * current_configs[torch.arange(n_chains), flip_indices]
            props = current_configs.scatter_add(1, flip_indices.unsqueeze(1), updates.unsqueeze(1))

            log_probs_prop = model.log_prob(props).real
            
            # Acceptance probability for all chains at once
            acceptance_prob = torch.exp(2 * (log_probs_prop - log_probs_current))
            
            # Decide which chains accept the move
            accept_mask = torch.rand(n_chains, device=device) < acceptance_prob
            
            # Update configurations and their log-probs
            current_configs[accept_mask] = props[accept_mask]
            log_probs_current[accept_mask] = log_probs_prop[accept_mask]

        # Sampling phase
        samples = torch.zeros(n_samples_per_chain, n_chains, n_spins, device=device)
        for i in range(n_samples_per_chain):
            for _ in range(step_interval):
                flip_indices = torch.randint(0, n_spins, (n_chains,), device=device)
                
                updates = -2 * current_configs[torch.arange(n_chains), flip_indices]
                props = current_configs.scatter_add(1, flip_indices.unsqueeze(1), updates.unsqueeze(1))
                
                log_probs_prop = model.log_prob(props).real
                acceptance_prob = torch.exp(2 * (log_probs_prop - log_probs_current))
                accept_mask = torch.rand(n_chains, device=device) < acceptance_prob
                
                current_configs[accept_mask] = props[accept_mask]
                log_probs_current[accept_mask] = log_probs_prop[accept_mask]
            
            samples[i] = current_configs

    # Reshape to combine all chains into one large sample batch
    return samples.view(-1, n_spins)


def local_energy_batch(model, samples, qubit_hamiltonian, device='cpu'):
    """
    Compute the local energy for a batch of samples.
    """
    n_samples = samples.shape[0]
    E_loc = torch.zeros(n_samples, dtype=torch.complex128, device=device)
    
    with torch.no_grad():
        log_psi = model.log_prob(samples)

        for term, coeff in qubit_hamiltonian.terms.items():
            if not term:  # Identity term
                E_loc += coeff
                continue

            samples_prime = samples.clone()
            phase = torch.ones(n_samples, dtype=torch.complex128, device=device)
            
            for idx, pauli in term:
                vals = samples[:, idx]
                if pauli == 'Z':
                    phase *= vals
                else:  # X or Y
                    samples_prime[:, idx] *= -1
                    if pauli == 'Y':
                        phase *= 1j * vals
            
            log_psi_prime = model.log_prob(samples_prime)
            amp_ratio = torch.exp(log_psi_prime - log_psi)
            E_loc += coeff * phase * amp_ratio
        
    return E_loc.real


def stochastic_reconfiguration_update(model, samples, qubit_ham, lr, reg, device='cpu'):
    """
    Updates the RBM parameters using the Stochastic Reconfiguration (SR) method,
    optimized for batch operations.
    """
    params = list(model.parameters())
    n_p = sum(p.numel() for p in params)
    n_samples = samples.shape[0]

    # Compute log-derivative vectors O_k (vectorized)
    # This requires per-sample gradients. The most efficient way is with functorch (torch.func.vmap),
    # but a loop is used here for compatibility with standard PyTorch.
    O_list = []
    log_psi_batch = model.log_prob(samples)
    for i in range(n_samples):
        model.zero_grad()
        # Backward pass for a single sample's log_prob
        log_psi_batch[i].real.backward(retain_graph=True)
        grad_vec = torch.cat([p.grad.view(-1) for p in params])
        O_list.append(grad_vec)
    O = torch.stack(O_list)  # Shape: [n_samples, n_params]

    O_conj = O.conj()
    
    # Compute covariance matrix S (vectorized)
    S = torch.einsum('ki,kj->ij', O_conj, O) / n_samples
    S -= torch.outer(O_conj.mean(dim=0), O.mean(dim=0))
    S += reg * torch.eye(n_p, dtype=torch.complex128, device=device)

    # Compute force vector F (vectorized)
    local_es = local_energy_batch(model, samples, qubit_ham, device=device).to(torch.complex128)
    F = torch.einsum('ki,k->i', O_conj, local_es) / n_samples
    F -= O_conj.mean(dim=0) * local_es.mean()
    
    # Solve linear system S * delta = F
    try:
        # Move S and F to CPU for torch.linalg.solve if they are on GPU
        S_cpu = S.cpu()
        F_cpu = F.cpu()
        delta = torch.linalg.solve(S_cpu, F_cpu).to(device)
        
        with torch.no_grad():
            idx = 0
            for p in params:
                num = p.numel()
                # Use the real part of the update
                update = delta[idx:idx+num].reshape(p.shape).real
                p -= lr * update
                idx += num
    except torch.linalg.LinAlgError:
        print("SR matrix is singular; skipping update.")