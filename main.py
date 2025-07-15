import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from models import RBM
from moleculars import get_pyscf_results, MOLECULE_DATA
from vmc_cal import metropolis_hastings_sampler, stochastic_reconfiguration_update, local_energy

def load_config(path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(path, 'r') as file:
        return yaml.safe_load(file)
    
def plot_results(results, seps, config):
    """Plots the final energy results."""
    molecule_name = config['molecule_name']
    basis = MOLECULE_DATA[molecule_name]['basis']
    
    plt.figure(figsize=(12, 8))
    methods = list(results.keys())
    
    if config['dissociation_curve']['enabled'] and len(seps) > 1:
        for method in methods:
            plt.plot(seps, results[method], marker='o', linestyle='-', label=method)
        plt.xlabel("Separation (Å)")
        plt.title(f"{molecule_name} Potential Energy Dissociation Curve ({basis})")
    else:
        values = [results[method][0] for method in methods]
        plt.bar(methods, values)
        plt.title(f"{molecule_name} Ground State Energy ({basis})")

    plt.ylabel("Energy (Hartree)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load configuration
    config = load_config()
    molecule_choice = config['molecule_name']
    vmc_params = config['vmc_params']
    rbm_params = config['rbm_params']

    if molecule_choice not in MOLECULE_DATA:
        print(f"Error: Molecule '{molecule_choice}' not found in MOLECULE_DATA.")
        exit()

    # Determine separation points for the calculation
    geom = MOLECULE_DATA[molecule_choice]['geometry']
    is_diatomic = len(geom) == 2
    
    if is_diatomic and config['dissociation_curve']['enabled']:
        curve_config = config['dissociation_curve']
        seps = np.linspace(curve_config['start_separation'], curve_config['end_separation'], curve_config['points'])
        # Calculate scaling factor from the equilibrium distance
        base_dist = np.linalg.norm(np.array(geom[0][1]) - np.array(geom[1][1]))
        scales = seps / base_dist
    else:
        seps = [np.linalg.norm(np.array(geom[0][1]) - np.array(geom[1][1]))] if is_diatomic else [0]
        scales = [1.0]

    # Store results
    results = {'HF': [], 'FCI': [], 'CCSD': [], 'CCSD(T)': [], 'RBM': []}

    for i, scale in enumerate(scales):
        label = f"{seps[i]:.3f} Å" if len(scales) > 1 else "Equilibrium Geometry"
        print(f"\n--- Calculating for {molecule_choice} @ {label} ---")

        # 1. Get ground truth energies and qubit Hamiltonian from PySCF
        mol, hf_e, fci_e, ccsd_e, ccsd_t_e, qham = get_pyscf_results(molecule_choice, scale)
        results['HF'].append(hf_e)
        results['FCI'].append(fci_e)
        results['CCSD'].append(ccsd_e)
        results['CCSD(T)'].append(ccsd_t_e)
        print("Classical quantum chemistry calculations complete.")

        # 2. Initialize the RBM model
        n_orbitals = mol.nao_nr() * 2  # Number of spin-orbitals
        n_hidden = int(n_orbitals * rbm_params['alpha'])
        rbm = RBM(n_orbitals, n_hidden)
        
        # 3. Train the RBM using VMC
        print(f"Starting RBM training for {vmc_params['n_epochs']} epochs...")
        for ep in range(vmc_params['n_epochs']):
            samples = metropolis_hastings_sampler(rbm, vmc_params['n_samples'], n_orbitals, vmc_params['burn_in_steps'])
            stochastic_reconfiguration_update(
                rbm, samples, qham, 
                lr=vmc_params['learning_rate'], 
                reg=vmc_params['sr_regularization']
            )
            if (ep + 1) % 10 == 0:
                print(f"  Epoch {ep+1}/{vmc_params['n_epochs']} complete.")
        
        print("RBM training finished.")

        # 4. Calculate final RBM energy
        final_samples = metropolis_hastings_sampler(rbm, vmc_params['n_samples'] * 2, n_orbitals, vmc_params['burn_in_steps'])
        local_energies = [local_energy(rbm, s, qham) for s in final_samples]
        stacked = torch.stack(local_energies).detach()
        rbm_e = stacked.mean().item()
        rbm_e_std = stacked.std().item() / np.sqrt(len(local_energies))
        results['RBM'].append(rbm_e)

        print("\n--- Results Summary ---")
        print(f"  Hartree-Fock: {hf_e:.6f} Ha")
        print(f"  FCI:          {fci_e:.6f} Ha")
        print(f"  CCSD:         {ccsd_e:.6f} Ha")
        print(f"  CCSD(T):      {ccsd_t_e:.6f} Ha")
        print(f"  RBM-VMC:      {rbm_e:.6f} ± {rbm_e_std:.6f} Ha")

    # 5. Plot all results
    plot_results(results, seps, config)