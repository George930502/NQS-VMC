import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from models_torch import RBM
from moleculars_torch import get_pyscf_results, MOLECULE_DATA
from vmc_cal_torch import parallel_metropolis_hastings_sampler, stochastic_reconfiguration_update, local_energy_batch

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

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if molecule_choice not in MOLECULE_DATA:
        print(f"Error: Molecule '{molecule_choice}' not found in MOLECULE_DATA.")
        exit()

    geom = MOLECULE_DATA[molecule_choice]['geometry']
    is_diatomic = len(geom) == 2
    
    if is_diatomic and config['dissociation_curve']['enabled']:
        curve_config = config['dissociation_curve']
        seps = np.linspace(curve_config['start_separation'], curve_config['end_separation'], curve_config['points'])
        base_dist = np.linalg.norm(np.array(geom[0][1]) - np.array(geom[1][1]))
        scales = seps / base_dist if base_dist > 0 else np.ones_like(seps)
    else:
        seps = [np.linalg.norm(np.array(geom[0][1]) - np.array(geom[1][1]))] if is_diatomic else [0]
        scales = [1.0]

    results = {'HF': [], 'FCI': [], 'CCSD': [], 'CCSD(T)': [], 'RBM': []}

    for i, scale in enumerate(scales):
        label = f"{seps[i]:.3f} Å" if len(scales) > 1 else "Equilibrium Geometry"
        print(f"\n--- Calculating for {molecule_choice} @ {label} ---")

        mol, hf_e, fci_e, ccsd_e, ccsd_t_e, qham = get_pyscf_results(molecule_choice, scale)
        results['HF'].append(hf_e)
        results['FCI'].append(fci_e)
        results['CCSD'].append(ccsd_e)
        results['CCSD(T)'].append(ccsd_t_e)
        print("Classical quantum chemistry calculations complete.")

        n_orbitals = mol.nao_nr() * 2
        n_hidden = int(n_orbitals * rbm_params['alpha'])
        rbm = RBM(n_orbitals, n_hidden, device=device)
        
        print(f"Starting RBM training for {vmc_params['n_epochs']} epochs...")
        n_chains = vmc_params['n_chains']
        n_samples_per_chain = vmc_params['n_samples'] // n_chains

        for ep in range(vmc_params['n_epochs']):
            samples = parallel_metropolis_hastings_sampler(
                rbm, n_samples_per_chain, n_chains, n_orbitals,
                vmc_params['burn_in_steps'], n_orbitals * vmc_params['step_intervals'], device=device
            )
            
            stochastic_reconfiguration_update(
                rbm, samples, qham, 
                lr=vmc_params['learning_rate'], 
                reg=vmc_params['sr_regularization'],
                device=device
            )

            if (ep + 1) % 10 == 0:
                # To evaluate energy, generate a fresh batch of samples
                eval_samples = parallel_metropolis_hastings_sampler(
                    rbm, n_samples_per_chain, n_chains, n_orbitals,
                    vmc_params['burn_in_steps'], n_orbitals * vmc_params['step_intervals'], device=device
                )
                eval_local_energies = local_energy_batch(rbm, eval_samples, qham, device)
                eval_mean = eval_local_energies.mean().item()
                eval_std = eval_local_energies.std(unbiased=False).item() / np.sqrt(len(eval_local_energies))
                
                print(f"  Epoch {ep+1}/{vmc_params['n_epochs']} complete. "
                      f"Current RBM Energy = {eval_mean:.6f} ± {eval_std:.6f} Ha")
        
        print("RBM training finished.")

        final_samples = parallel_metropolis_hastings_sampler(
            rbm, n_samples_per_chain, n_chains, n_orbitals,
            vmc_params['burn_in_steps'], n_orbitals * vmc_params['step_intervals'], device=device
        )
        local_energies = local_energy_batch(rbm, final_samples, qham, device)
        rbm_e = local_energies.mean().item()
        rbm_e_std = local_energies.std().item() / np.sqrt(len(local_energies))
        results['RBM'].append(rbm_e)

        print("\n--- Results Summary ---")
        print(f"  Hartree-Fock: {hf_e:.6f} Ha")
        print(f"  FCI:          {fci_e:.6f} Ha")
        print(f"  CCSD:         {ccsd_e:.6f} Ha")
        print(f"  CCSD(T):      {ccsd_t_e:.6f} Ha")
        print(f"  RBM-VMC:      {rbm_e:.6f} ± {rbm_e_std:.6f} Ha")

    plot_results(results, seps, config)