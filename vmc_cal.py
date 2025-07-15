import torch

def local_energy(model, sigma, qubit_hamiltonian):
    """
    Compute the local energy E_loc(sigma) = <sigma|H|Psi>/<sigma|Psi>
    using the qubit_hamiltonian (an OpenFermion QubitOperator).
    """
    E_loc = 0+0j
    log_psi = model.log_prob(sigma)

    for term, coeff in qubit_hamiltonian.terms.items():
        if not term: # Identity term
            E_loc += coeff
            continue

        sigma_prime = sigma.clone()
        phase = 1+0j
        
        # Apply each Pauli in the term
        for idx, pauli in term:
            val = sigma[idx].item()
            if pauli == 'Z':
                phase *= (1 if val > 0 else -1)
            else: # X or Y
                sigma_prime[idx] = -sigma_prime[idx]
                if pauli == 'Y':
                    phase *= (1j if val > 0 else -1j)
        
        log_psi_prime = model.log_prob(sigma_prime)
        amp_ratio = torch.exp(log_psi_prime - log_psi)
        E_loc += coeff * phase * amp_ratio
        
    return E_loc.real

def metropolis_hastings_sampler(model, n_samples, n_spins, burn_in):
    """
    Generates spin configurations from the RBM probability distribution using Metropolis-Hastings.
    """
    samples = []
    # Start with a random spin configuration
    current = torch.randint(0, 2, (n_spins,)).float() * 2 - 1
    
    # Burn-in phase to reach equilibrium
    for _ in range(burn_in):
        prop = current.clone()
        flip_idx = torch.randint(0, n_spins, (1,)).item()
        prop[flip_idx] *= -1
        
        log_prob_current = model.log_prob(current).real
        log_prob_proposal = model.log_prob(prop).real
        
        acceptance_prob = torch.exp(2 * (log_prob_proposal - log_prob_current))
        if torch.rand(1) < acceptance_prob:
            current = prop
            
    # Sampling phase
    for _ in range(n_samples):
        prop = current.clone()
        flip_idx = torch.randint(0, n_spins, (1,)).item()
        prop[flip_idx] *= -1
        
        log_prob_current = model.log_prob(current).real
        log_prob_proposal = model.log_prob(prop).real
        
        acceptance_prob = torch.exp(2 * (log_prob_proposal - log_prob_current))
        if torch.rand(1) < acceptance_prob:
            current = prop
        samples.append(current)
        
    return torch.stack(samples)

def stochastic_reconfiguration_update(model, samples, qubit_ham, lr, reg):
    """
    Updates the RBM parameters using the Stochastic Reconfiguration (SR) method.
    """
    params = list(model.parameters())
    n_p = sum(p.numel() for p in params)

    # Compute log-derivative vectors O_k
    O_list = []
    for s in samples:
        model.zero_grad()
        log_psi = model.log_prob(s).real
        log_psi.backward()
        grad_vec = torch.cat([p.grad.view(-1) for p in params])
        O_list.append(grad_vec)
    O = torch.stack(O_list) # shape (N_samples, n_params)

    O_conj = O.conj()
    
    # Compute covariance matrix S
    S = torch.einsum('ki,kj->ij', O_conj, O) / len(samples)
    S -= torch.outer(O_conj.mean(dim=0), O.mean(dim=0))
    S += reg * torch.eye(n_p, dtype=torch.cfloat)

    # Compute force vector F
    local_es = torch.tensor([local_energy(model, s, qubit_ham) for s in samples], dtype=torch.cfloat)
    F = torch.einsum('ki,k->i', O_conj, local_es) / len(samples)
    F -= O_conj.mean(dim=0) * local_es.mean()
    
    # Solve linear system S * delta = F and update parameters
    try:
        delta = torch.linalg.solve(S, F)
        with torch.no_grad():
            idx = 0
            for p in params:
                num = p.numel()
                # Use the real part of the update, as parameters are complex but the update should be in a real direction
                update = delta[idx:idx+num].reshape(p.shape).real
                p -= lr * update
                idx += num
    except torch.linalg.LinAlgError:
        print("SR matrix is singular; skipping update.")