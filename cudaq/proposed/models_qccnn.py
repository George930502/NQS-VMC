import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

def make_qnode(n_qubits, n_layers, dev):
    """Creates a quantum node (quantum circuit) for the QConv layer."""
    @qml.qnode(dev, interface="torch", diff_method="adjoint")

    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation='Y')
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer * n_qubits + i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        observable = qml.PauliZ(0)
        for i in range(1, n_qubits):
            observable = observable @ qml.PauliZ(i)
        return qml.expval(observable)
    
    weight_shapes = {"weights": (n_layers * n_qubits,)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)

class QConv1d(nn.Module):
    """1D Quantum Convolutional Layer"""
    def __init__(self, out_channels, kernel_size, n_layers=4, stride=1, pennylane_device='default.qubit'):
        super(QConv1d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_layers = n_layers

        try:
            self.dev = qml.device(pennylane_device, wires=self.kernel_size)
        except qml.DeviceError:
            print(f"PennyLane device '{pennylane_device}' not found. Falling back to 'default.qubit'.")
            self.dev = qml.device("default.qubit", wires=self.kernel_size)

        self.quantum_layers = nn.ModuleList([
            make_qnode(self.kernel_size, n_layers, self.dev) for _ in range(out_channels)
        ])

    def forward(self, x):
        batch_size, _, length = x.shape
        out_length = (length - self.kernel_size) // self.stride + 1
        output = torch.zeros(batch_size, self.out_channels, out_length, device=x.device)

        for i in range(out_length):
            start = i * self.stride
            end = start + self.kernel_size
            patch = x[:, :, start:end].reshape(batch_size, -1)
            for k, layer in enumerate(self.quantum_layers):
                output[:, k, i] = layer(patch).squeeze()
        return output

class QCCNN(nn.Module):
    """
    Hybrid Quantum-Classical Convolutional Neural Network for modeling the wave function.
    """
    def __init__(self, n_spins, qccnn_params, torch_device='cpu', pennylane_device='default.qubit'):
        super(QCCNN, self).__init__()
        self.n_spins = n_spins
        self.torch_device = torch_device

        conv_layers = []
        current_spins = n_spins
        in_channels = 1
        
        for i in range(qccnn_params['num_conv_layers']):
            kernel_size = qccnn_params['kernel_size']
            out_channels = qccnn_params['out_channels']
            n_q_layers = qccnn_params['n_layers_quantum']
            
            if current_spins < kernel_size:
                print(f"Warning: Input size ({current_spins}) is smaller than kernel size ({kernel_size}). "
                      f"Stopping creation of convolutional layers at layer {i}.")
                break

            conv_layers.append(QConv1d(out_channels, kernel_size, n_layers=n_q_layers, pennylane_device=pennylane_device))
            current_spins = (current_spins - kernel_size) + 1
            in_channels = out_channels

        self.conv_net = nn.Sequential(*conv_layers)

        final_flat_size = current_spins * in_channels
        
        ffnn_hidden_size = qccnn_params['ffnn_hidden_size']
        self.ffnn = nn.Sequential(
            nn.Linear(final_flat_size, ffnn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffnn_hidden_size, 2)
        )
        self.to(self.torch_device)

    def log_prob(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculates the logarithm of the wave function amplitude (log_psi).
        """
        x = (sigma.float() + 1.0) / 2.0
        x = x.view(x.shape[0], 1, self.n_spins)
        #print(f"Input shape to convolution: {x.shape}")
        x = self.conv_net(x)
        #print(f"Output shape after convolution: {x.shape}")
        x = x.view(x.shape[0], -1)
        ffnn_out = self.ffnn(x)
        #print(f"Output shape after FFNN: {ffnn_out.shape}")
        log_psi = ffnn_out[:, 0] + 1j * ffnn_out[:, 1]
        return log_psi
    

if __name__ == "__main__":
    # Example usage
    n_spins = 8
    qccnn_params = {
        'num_conv_layers': 3,
        'kernel_size': 3,
        'out_channels': 1,
        'n_layers_quantum': 3,
        'ffnn_hidden_size': 16
    }
    
    model = QCCNN(n_spins, qccnn_params, torch_device='cuda', pennylane_device='lightning.gpu')
    print(model)
    
    # Test with a random configuration
    sigma = torch.randint(0, 2, (5, n_spins)) * 2 - 1  # Random spins in {-1, 1}
    model.eval()
    with torch.no_grad():
        log_psi = model.log_prob(sigma.to(model.torch_device))
        print("Log probability:", log_psi)