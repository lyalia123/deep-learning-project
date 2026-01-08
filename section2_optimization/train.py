import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
import seaborn as sns

from model import MLP
from losses import cross_entropy, cross_entropy_backward
from optimizers import SGD, RMSprop, Adam, GradientDescent
from gradient_checking import gradient_check, test_gradient_check
from utils import accuracy, create_minibatches, one_hot_encode

# Set random seeds for reproducibility
np.random.seed(42)

# ============================================================================
# 1. GRADIENT CHECKING DEMONSTRATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 2.2: GRADIENT CHECKING")
print("="*80)

# Test gradient checking
print("\n1. Testing gradient checking on small synthetic dataset...")
passed, max_error = test_gradient_check()

if passed:
    print("✓ Gradient checking passed! Backpropagation implementation is correct.")
else:
    print("✗ Gradient checking failed! Check backpropagation implementation.")

# ============================================================================
# 2. DATASET PREPARATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 2.3: OPTIMIZATION ALGORITHMS COMPARISON")
print("="*80)

print("\n2. Creating datasets for optimization comparison...")

# Create three datasets of increasing complexity
datasets = []

# Dataset 1: Simple linearly separable
X1, y1 = make_classification(n_samples=500, n_features=2, n_informative=2,
                            n_redundant=0, n_clusters_per_class=1,
                            flip_y=0.01, random_state=1)
y1 = one_hot_encode(y1, 2)
datasets.append(("Linear", X1, y1))

# Dataset 2: Moons (non-linear)
X2, y2 = make_moons(n_samples=500, noise=0.1, random_state=2)
y2 = one_hot_encode(y2, 2)
datasets.append(("Moons", X2, y2))

# Dataset 3: Complex non-linear
X3, y3 = make_classification(n_samples=500, n_features=2, n_informative=2,
                            n_redundant=0, n_clusters_per_class=2,
                            flip_y=0.1, random_state=3)
y3 = one_hot_encode(y3, 2)
datasets.append(("Complex", X3, y3))

# ============================================================================
# 3. TRAINING FUNCTION WITH DIFFERENT OPTIMIZERS
# ============================================================================

def train_experiment(optimizer_class, optimizer_name, X, y, 
                     hidden_dim=32, epochs=100, lr=0.01, **optimizer_kwargs):
    """
    Train MLP with given optimizer and return training history
    """
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    
    # Initialize model and optimizer
    model = MLP(input_dim, hidden_dim, output_dim, activation="relu")
    optimizer = optimizer_class(lr=lr, **optimizer_kwargs)
    
    # Training history
    history = {
        'loss': [],
        'accuracy': [],
        'grad_norm': [],
        'param_norm': []
    }
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_grad_norm = 0
        
        # Mini-batch training
        for X_batch, y_batch in create_minibatches(X, y, batch_size=32, shuffle=True):
            # Forward pass
            pred = model.predict_proba(X_batch)
            loss = cross_entropy(pred, y_batch)
            
            # Backward pass
            grad_loss = cross_entropy_backward(pred, y_batch)
            model.backward(grad_loss)
            
            # Update parameters
            optimizer.step(model)
            
            # Track statistics
            epoch_loss += loss
            grads = model.gradients()
            grad_norm = sum(np.linalg.norm(g) for g in grads.values())
            epoch_grad_norm += grad_norm
            
            # Reset gradients
            model.zero_grad()
        
        # Compute epoch statistics
        avg_loss = epoch_loss / (len(X) / 32)
        avg_grad_norm = epoch_grad_norm / (len(X) / 32)
        
        # Compute accuracy on full training set
        train_pred = model.predict(X)
        train_true = np.argmax(y, axis=1)
        train_acc = np.mean(train_pred == train_true)
        
        # Compute parameter norms
        params = model.parameters()
        param_norm = sum(np.linalg.norm(p) for p in params.values())
        
        # Store history
        history['loss'].append(avg_loss)
        history['accuracy'].append(train_acc)
        history['grad_norm'].append(avg_grad_norm)
        history['param_norm'].append(param_norm)
        
        if epoch % 20 == 0:
            print(f"  {optimizer_name}: Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                  f"Acc: {train_acc:.3f} | Grad Norm: {avg_grad_norm:.2e}")
    
    return history, model

# ============================================================================
# 4. RUN OPTIMIZATION EXPERIMENTS
# ============================================================================

print("\n3. Running optimization experiments on three datasets...")

# Define optimizers to compare
optimizers = [
    (GradientDescent, "GD", {"lr": 0.1}),
    (SGD, "SGD", {"lr": 0.1, "momentum": 0.9}),
    (RMSprop, "RMSprop", {"lr": 0.01}),
    (Adam, "Adam", {"lr": 0.01}),
]

results = {}

for dataset_name, X, y in datasets:
    print(f"\n  Dataset: {dataset_name}")
    print("  " + "-" * 50)
    
    dataset_results = {}
    
    for optimizer_class, optimizer_name, kwargs in optimizers:
        print(f"\n    Training with {optimizer_name}...")
        history, model = train_experiment(
            optimizer_class, optimizer_name, X, y,
            hidden_dim=32, epochs=100, **kwargs
        )
        dataset_results[optimizer_name] = history
    
    results[dataset_name] = dataset_results

# ============================================================================
# 5. VISUALIZATION AND ANALYSIS
# ============================================================================

import matplotlib.pyplot as plt

# Global style for better readability (PDF friendly)
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 11
})

fig = plt.figure(figsize=(18, 12))

# Color scheme for optimizers
colors = {
    'GD': 'blue',
    'SGD': 'green',
    'RMSprop': 'orange',
    'Adam': 'red'
}

datasets_order = list(results.keys())

for row_idx, dataset_name in enumerate(datasets_order):
    dataset_results = results[dataset_name]

    # ---- LOSS ----
    ax = plt.subplot(3, 4, row_idx * 4 + 1)
    for opt_name, history in dataset_results.items():
        ax.plot(history['loss'], color=colors[opt_name], linewidth=2, label=opt_name)
    ax.set_title(dataset_name)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # ---- ACCURACY ----
    ax = plt.subplot(3, 4, row_idx * 4 + 2)
    for opt_name, history in dataset_results.items():
        ax.plot(history['accuracy'], color=colors[opt_name], linewidth=2)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # ---- GRADIENT NORM ----
    ax = plt.subplot(3, 4, row_idx * 4 + 3)
    for opt_name, history in dataset_results.items():
        ax.plot(history['grad_norm'], color=colors[opt_name], linewidth=2)
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # ---- PARAMETER NORM ----
    ax = plt.subplot(3, 4, row_idx * 4 + 4)
    for opt_name, history in dataset_results.items():
        ax.plot(history['param_norm'], color=colors[opt_name], linewidth=2)
    ax.set_ylabel("Parameter L2 Norm")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

# Shared legend (single legend for entire figure)
handles = [
    plt.Line2D([0], [0], color=colors[name], lw=3)
    for name in colors
]
labels = list(colors.keys())

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False
)

# Main title


plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.subplots_adjust(
    hspace=0.55,  # расстояние между строками (верх–низ)
    wspace=0.25   # расстояние между столбцами (лево–право)
)
plt.show()

# ============================================================================
# 6. LEARNING RATE SCHEDULE EXPERIMENT
# ============================================================================

print("\n5. Learning rate schedule experiment...")

# Define learning rate schedules
def constant_schedule(epoch, lr0=0.1):
    return lr0

def step_decay_schedule(epoch, lr0=0.1, drop=0.5, epochs_drop=30):
    return lr0 * (drop ** (epoch // epochs_drop))

def exponential_decay_schedule(epoch, lr0=0.1, k=0.01):
    return lr0 * np.exp(-k * epoch)

def cosine_annealing_schedule(epoch, lr0=0.1, T_max=100):
    return lr0 * 0.5 * (1 + np.cos(np.pi * epoch / T_max))

# Test schedules
schedules = [
    ("Constant", constant_schedule),
    ("Step Decay", step_decay_schedule),
    ("Exponential", exponential_decay_schedule),
    ("Cosine", cosine_annealing_schedule),
]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot learning rate schedules
epochs = 100
for schedule_name, schedule_fn in schedules:
    lrs = [schedule_fn(epoch) for epoch in range(epochs)]
    axes[0].plot(lrs, label=schedule_name, linewidth=2)

axes[0].set_title('Learning Rate Schedules')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Learning Rate')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Train with different schedules
X, y = datasets[1][1], datasets[1][2]  # Use Moons dataset

schedule_results = {}
for schedule_name, schedule_fn in schedules:
    print(f"\n  Training with {schedule_name} schedule...")
    
    model = MLP(X.shape[1], 32, y.shape[1], activation="relu")
    optimizer = SGD(lr=0.1, momentum=0.9)
    
    losses = []
    for epoch in range(epochs):
        # Update learning rate
        optimizer.lr = schedule_fn(epoch)
        
        epoch_loss = 0
        for X_batch, y_batch in create_minibatches(X, y, batch_size=32, shuffle=True):
            pred = model.predict_proba(X_batch)
            loss = cross_entropy(pred, y_batch)
            grad_loss = cross_entropy_backward(pred, y_batch)
            
            model.backward(grad_loss)
            optimizer.step(model)
            model.zero_grad()
            
            epoch_loss += loss
        
        losses.append(epoch_loss / (len(X) / 32))
    
    schedule_results[schedule_name] = losses

# Plot loss with different schedules
for schedule_name, losses in schedule_results.items():
    axes[1].plot(losses, label=schedule_name, linewidth=2)

axes[1].set_title('Training Loss with Different LR Schedules (SGD)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 7. FINAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS AND CONCLUSIONS")
print("="*80)

print("""
1. GRADIENT CHECKING:
   - Gradient checking verifies that our backpropagation implementation is correct.
   - Relative errors should be < 1e-7 for numerical stability.
   - This step is crucial before running any optimization experiments.

2. OPTIMIZER COMPARISON:
   - GD (Gradient Descent): Simple but can be slow and may oscillate.
   - SGD with momentum: Faster convergence, reduces oscillations.
   - RMSprop: Adapts learning rate per parameter, good for non-stationary objectives.
   - Adam: Combines momentum and adaptive learning rates, generally fastest.

3. LEARNING RATE SCHEDULES:
   - Constant: Simple but may not converge to optimum.
   - Step decay: Good for plateaus, mimics manual adjustment.
   - Exponential: Smooth decay, good for convex problems.
   - Cosine annealing: Helps escape local minima, good for deep networks.

4. OBSERVATIONS:
   - Adam typically achieves fastest convergence across all datasets.
   - Gradient norms decay faster with adaptive methods (RMSprop, Adam).
   - Parameter norms stabilize with adaptive optimizers.
   - Choice of optimizer depends on problem complexity and dataset.

5. RECOMMENDATIONS:
   - Always start with Adam for general problems.
   - Use gradient checking to verify implementations.
   - Experiment with learning rate schedules for final fine-tuning.
   - Monitor gradient norms to detect vanishing/exploding gradients.
""")

print("\n" + "="*80)
print("SECTION 2 COMPLETE")
print("="*80)