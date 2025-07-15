# Quantum Many-Body Problem with RBM

This project implements the methods described in the paper [**Fermionic neural-network states for ab-initio electronic structure**](https://www.nature.com/articles/s41467-020-15724-9), using a **Restricted Boltzmann Machine (RBM)** as a variational ansatz to approximate the ground state of interacting electronic systems.

---

## 🧠 Overview

The aim of this project is to solve quantum many-body problems in electronic structure theory using neural-network quantum states (NNQS), specifically the RBM architecture. We target the ground-state energy of small molecules and benchmark our results against traditional quantum chemistry methods.

---

## ✨ Features

- 🧪 Compute **ground-state energies** for molecules like **H₂**, **LiH**, **C₂**, etc.
- 📉 Generate **potential energy dissociation curves** for diatomic molecules.
- ⚖️ Compare RBM results with standard quantum chemistry baselines:
  - **Hartree-Fock (HF)**
  - **Full Configuration Interaction (FCI)**
  - **Coupled Cluster Singles and Doubles (CCSD)**
  - **CCSD(T)**
- 🔧 Easy experiment configuration via `config.yaml`:
  - molecule type, basis set, bond length range, RBM parameters, etc.
- 💾 Includes support for loading basis set and molecular integrals from quantum chemistry packages (e.g., PySCF).

---

## 🛠 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/George930502/NQS-VMC.git
cd VMC-NQS
pip install -r requirements.txt
```

---

## ⚙️ Configuration
All simulation parameters are defined in ```config.yaml```. Here's an example:

```bash
# Specify the target molecule.
# Available options: 'H2', 'LiH', 'C2', 'N2', 'NH3', 'H2O_STO-3G', 'H2O_6-31G'
molecule_name: 'H2'

# Configuration for potential energy surface scan (for diatomic molecules)
dissociation_curve:
  enabled: true
  start_separation: 0.5   # in Angstrom
  end_separation: 2.5     # in Angstrom
  points: 15              # Number of points to compute

# Parameters for the Variational Monte Carlo (VMC) simulation
vmc_params:
  n_epochs: 100
  n_samples: 1000
  learning_rate: 0.05
  burn_in_steps: 100  # Burn-in for the Metropolis-Hastings sampler
  sr_regularization: 0.01 # Regularization for Stochastic Reconfiguration

# Parameters for the RBM model
rbm_params:
  # The number of hidden units will be n_orbitals * alpha
  alpha: 1
```

---

## ▶️ Running the Simulation

To run the simulation and generate the energy data:

```bash
python main.py
```

---

## 📚 Reference

If you use this codebase or are interested in the theory, please refer to the original paper: [**Fermionic neural-network states for ab-initio electronic structure**](https://www.nature.com/articles/s41467-020-15724-9)

---
## 📌 License

MIT License. See ```LICENSE``` file for details.