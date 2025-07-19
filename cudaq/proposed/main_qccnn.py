import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from models_qccnn import QCCNN
from moleculars_qccnn import get_pyscf_results, MOLECULE_DATA
from vmc_cal_qccnn import parallel_metropolis_hastings_sampler, stochastic_reconfiguration_update, local_energy_batch

def load_config(path="config_qccnn.yaml"):
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
    config = load_config()
    molecule_choice = config['molecule_name']
    vmc_params = config['vmc_params']
    qccnn_params = config['qccnn_params']

    # Define separate devices for PyTorch and PennyLane
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pennylane_device = "lightning.gpu" if torch.cuda.is_available() else "default.qubit"
    
    print(f"Using PyTorch device: {torch_device}")
    print(f"Using PennyLane device: {pennylane_device}")

    if molecule_choice not in MOLECULE_DATA:
        raise ValueError(f"Molecule '{molecule_choice}' not found.")

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

    results = {'HF': [], 'FCI': [], 'CCSD': [], 'CCSD(T)': [], 'QCCNN': []}

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
        print(f"Number of orbitals: {n_orbitals}")
        
        # Instantiate the model with separate device arguments
        model = QCCNN(n_spins=n_orbitals, 
                      qccnn_params=qccnn_params, 
                      torch_device=torch_device, 
                      pennylane_device=pennylane_device)
        
        print(f"Starting QCCNN training for {vmc_params['n_epochs']} epochs...")
        n_chains = vmc_params['n_chains']
        n_samples_per_chain = vmc_params['n_samples'] // n_chains

        for ep in range(vmc_params['n_epochs']):
            samples = parallel_metropolis_hastings_sampler(
                model, n_samples_per_chain, n_chains, n_orbitals,
                vmc_params['burn_in_steps'], n_orbitals * vmc_params['step_intervals'], device=torch_device
            )
            
            stochastic_reconfiguration_update(
                model, samples, qham, 
                lr=vmc_params['learning_rate'], 
                reg=vmc_params['sr_regularization'],
                device=torch_device
            )

            if (ep + 1) % 1 == 0:
                eval_samples = parallel_metropolis_hastings_sampler(
                    model, n_samples_per_chain, n_chains, n_orbitals,
                    vmc_params['burn_in_steps'], n_orbitals * vmc_params['step_intervals'], device=torch_device
                )
                eval_local_energies = local_energy_batch(model, eval_samples, qham, torch_device)
                eval_mean = eval_local_energies.mean().item()
                eval_std = eval_local_energies.std(unbiased=False).item() / np.sqrt(len(eval_local_energies))
                
                print(f"  Epoch {ep+1}/{vmc_params['n_epochs']} | QCCNN Energy = {eval_mean:.6f} ± {eval_std:.6f} Ha")
        
        print("QCCNN training finished.")

        final_samples = parallel_metropolis_hastings_sampler(
            model, n_samples_per_chain, n_chains, n_orbitals,
            vmc_params['burn_in_steps'], n_orbitals * vmc_params['step_intervals'], device=torch_device
        )
        local_energies = local_energy_batch(model, final_samples, qham, torch_device)
        qccnn_e = local_energies.mean().item()
        qccnn_e_std = local_energies.std().item() / np.sqrt(len(local_energies))
        results['QCCNN'].append(qccnn_e)

        print("\n--- Results Summary ---")
        print(f"  Hartree-Fock: {hf_e:.6f} Ha")
        print(f"  FCI:          {fci_e:.6f} Ha")
        print(f"  CCSD:         {ccsd_e:.6f} Ha")
        print(f"  CCSD(T):      {ccsd_t_e:.6f} Ha")
        print(f"  QCCNN-VMC:    {qccnn_e:.6f} ± {qccnn_e_std:.6f} Ha")

    plot_results(results, seps, config)