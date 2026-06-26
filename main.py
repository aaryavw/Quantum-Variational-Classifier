import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. GENERATE AND CLEAN DATA
# ==========================================
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)  # more samples helps generalization

# FIX: scale straight into [-pi, pi] with NO clipping.
# The old code used StandardScaler then clipped to [-1, 1] before multiplying
# by pi. That clip threw away almost all the spread in the data (anything
# beyond ~1 std dev got squashed to the same angle) -- this alone was likely
# the single biggest reason accuracy was so bad.
scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X = scaler.fit_transform(X)

# Convert labels from {0, 1} to {-1, 1} for quantum measurement
y = y * 2 - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. DEFINE THE QUANTUM CIRCUIT
# ==========================================
num_qubits = 2
num_layers = 6  # FIX: more layers -> more expressive circuit (was 3)
dev = qml.device("default.qubit", wires=num_qubits)

def quantum_layer(weights, wires):
    # FIX: each qubit now gets a full RX-RY-RZ rotation (3 free angles)
    # instead of just one rotation each. One rotation per qubit per layer
    # can't represent an arbitrary single-qubit state -- this was capping
    # the model's capacity hard.
    qml.RX(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[0])
    qml.RZ(weights[2], wires=wires[0])
    qml.RX(weights[3], wires=wires[1])
    qml.RY(weights[4], wires=wires[1])
    qml.RZ(weights[5], wires=wires[1])
    # Entangle in both directions so info flows both ways
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[0]])

@qml.qnode(dev)
def quantum_circuit(weights, x):
    for layer_weights in weights:
        # FIX: "data re-uploading" -- re-encode the classical input at every
        # layer, not just once at the start. This is the standard trick used
        # to get real expressivity out of a circuit with only 2 qubits.
        qml.AngleEmbedding(x, wires=range(num_qubits), rotation='X')
        quantum_layer(layer_weights, wires=range(num_qubits))

    # FIX: read out BOTH qubits instead of discarding qubit 1's information
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

# ==========================================
# 3. DEFINE LOSS AND BIAS
# ==========================================
def variational_classifier(weights, bias, x):
    q0, q1 = quantum_circuit(weights, x)
    # Combine both qubit measurements (simple average) + classical bias
    return 0.5 * (q0 + q1) + bias

def loss_function(weights, bias, X, y):
    predictions = np.stack([variational_classifier(weights, bias, x) for x in X])
    return np.mean((predictions - y) ** 2)

# ==========================================
# 4. TRAINING THE MODEL
# ==========================================
np.random.seed(42)
# FIX: weights initialized with a much wider spread (was 0.01*randn, which
# starts the circuit almost as the identity -> near-zero gradients).
weights_init = np.random.uniform(low=0, high=2 * np.pi, size=(num_layers, 6), requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

# FIX: Adam adapts its step size per parameter and converges far more
# reliably than plain gradient descent on this kind of bumpy loss surface.
opt = qml.AdamOptimizer(stepsize=0.1)

batch_size = 15  # slightly larger batch -> less noisy gradient estimate
weights = weights_init
bias = bias_init

num_epochs = 60  # Adam converges faster, so fewer epochs are needed

print("Starting Training...")
print("--------------------")
for it in range(num_epochs):
    batch_index = np.random.randint(0, len(X_train), (batch_size,))
    X_train_batch = X_train[batch_index]
    y_train_batch = y_train[batch_index]

    weights, bias = opt.step(lambda w, b: loss_function(w, b, X_train_batch, y_train_batch), weights, bias)

    predictions = np.array([np.sign(variational_classifier(weights, bias, x)) for x in X_train])
    accuracy = np.mean(predictions == y_train)

    print(f"Epoch {it+1:2d} | Loss: {loss_function(weights, bias, X_train, y_train):.4f} | Accuracy: {accuracy*100:.1f}%")

# ==========================================
# 5. TESTING AND EVALUATION
# ==========================================
test_predictions = np.array([np.sign(variational_classifier(weights, bias, x)) for x in X_test])
test_accuracy = np.mean(test_predictions == y_test)
print("--------------------")
print(f"Final Test Accuracy: {test_accuracy*100:.1f}%")
