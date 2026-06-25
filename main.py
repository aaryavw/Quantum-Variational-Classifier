import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. GENERATE AND CLEAN DATA
# ==========================================
# We use a toy dataset: two interleaving half-circles (moons)
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# Scale features to fit nicely into quantum states (mean=0, variance=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Quantum states rely on angles. Let's normalize data between [-pi, pi]
X = np.clip(X, -1, 1) * np.pi

# Convert labels from {0, 1} to {-1, 1} which is standard for quantum measurement
y = y * 2 - 1

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ==========================================
# 2. DEFINE THE QUANTUM CIRCUIT
# ==========================================
# We have 2 features, so we use 2 qubits
num_qubits = 2
dev = qml.device("default.qubit", wires=num_qubits)

# Define the layer structure of our quantum neural network
def quantum_layer(weights, wires):
    # Rotate qubits based on weights
    qml.RX(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    # Entangle the qubits together
    qml.CNOT(wires=[wires[0], wires[1]])

@qml.qnode(dev)
def quantum_circuit(weights, x):
    # Step A: Encode classical data 'x' into the quantum circuit
    qml.AngleEmbedding(x, wires=range(num_qubits), rotation='X')
    
    # Step B: Apply the trainable variational layers
    for weight in weights:
        quantum_layer(weight, wires=range(num_qubits))
        
    # Step C: Measure the first qubit (returns a value between -1 and 1)
    return qml.expval(qml.PauliZ(0))


# ==========================================
# 3. DEFINE LOSS AND BIAS
# ==========================================
# A classical bias term added to the quantum output
def variational_classifier(weights, bias, x):
    return quantum_circuit(weights, x) + bias

# Mean Squared Error Loss function
def loss_function(weights, bias, X, y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return np.mean((np.array(predictions) - y) ** 2)


# ==========================================
# 4. TRAINING THE MODEL
# ==========================================
# Define network structure: 3 layers, each needing 2 weight parameters
num_layers = 3
np.random.seed(42)
weights_init = 0.01 * np.random.randn(num_layers, num_qubits, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

# Use PennyLane's built-in Gradient Descent Optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.05)
batch_size = 10

weights = weights_init
bias = bias_init

print("Starting Training...")
print("--------------------")

for it in range(100):
    # Shuffle and batch the data for faster training
    batch_index = np.random.randint(0, len(X_train), (batch_size,))
    X_train_batch = X_train[batch_index]
    y_train_batch = y_train[batch_index]
    
    # Update weights and bias using the optimizer
    weights, bias = opt.step(lambda w, b: loss_function(w, b, X_train_batch, y_train_batch), weights, bias)
    
    # Calculate overall accuracy
    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X_train]
    accuracy = np.mean(predictions == y_train)
    
    print(f"Epoch {it+1:2d} | Loss: {loss_function(weights, bias, X_train, y_train):.4f} | Accuracy: {accuracy*100:.1f}%")


# ==========================================
# 5. TESTING AND EVALUATION
# ==========================================
test_predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X_test]
test_accuracy = np.mean(test_predictions == y_test)
print("--------------------")
print(f"Final Test Accuracy: {test_accuracy*100:.1f}%")
