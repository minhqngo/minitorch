# minitorch

This is a mini, torch-like deep learning library. The goal is that you can use syntax similar to PyTorch to build deep learning model using this library. This project starting points is the [minitorch exercises](https://github.com/minitorch/minitorch). However, after completing the exercises, I want to turn it into a functional deep learning library that can utilize GPU for model training, and I also want to refactor the code to make it clearer. This project is my attempt to do so.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Tensor Operations

```python
import minitorch

# Create tensors with different backends
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)  # If CUDA is available

# Create a tensor
x = minitorch.tensor([1, 2, 3, 4], backend=FastTensorBackend)
y = minitorch.tensor([5, 6, 7, 8], backend=FastTensorBackend)

# Perform operations
z = x + y
result = z.sum()
```

### Building Neural Networks

Create custom models by subclassing `minitorch.Module`:

```python
class Network(minitorch.Module):
    def __init__(self, backend):
        super().__init__()
        self.fc1 = minitorch.Linear(784, 128, backend=backend)
        self.fc2 = minitorch.Linear(128, 10, backend=backend)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = minitorch.dropout(x, 0.2, not self.training)
        x = self.fc2(x)
        return minitorch.logsoftmax(x, dim=1)
```

### Convolutional Neural Networks

```python
class CNN(minitorch.Module):
    def __init__(self, backend):
        super().__init__()
        self.conv1 = minitorch.Conv2d(in_channels=1, out_channels=6, kernel=(5, 5), stride=1, backend=backend)
        self.conv2 = minitorch.Conv2d(in_channels=6, out_channels=16, kernel=(5, 5), stride=1, backend=backend)
        self.fc1 = minitorch.Linear(16 * 4 * 4, 120, backend=backend)
        self.fc2 = minitorch.Linear(120, 10, backend=backend)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x).relu()
        x = minitorch.avgpool2d(x, kernel=(2, 2), stride=(2, 2))
        x = self.conv2(x).relu()
        x = minitorch.avgpool2d(x, kernel=(2, 2), stride=(2, 2))
        x = x.view(batch_size, -1)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x
```

### Training Loop

```python
# Initialize model and optimizer
model = Network(backend=FastTensorBackend)
optimizer = minitorch.RMSProp(model.parameters(), lr=0.01)

# Training
model.train()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = minitorch.nll_loss(output, y_batch)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
for X_batch, y_batch in val_loader:
    output = model(X_batch)
    predictions = minitorch.argmax(output, dim=1)
```

### Data Loading

```python
from minitorch.datasets import mnist
from minitorch.dataloader import DataLoader

# Load dataset
mnist_train = mnist.MNISTDataset("/path/to/data", train=True)

# Create dataloader
train_loader = DataLoader(
    mnist_train,
    batch_size=32,
    shuffle=True,
    backend=FastTensorBackend,
    transform=lambda x: x.astype(np.float64) / 255.0
)
```

### GPU Acceleration

```python
import numba

# Check CUDA availability and use GPU backend
if numba.cuda.is_available():
    backend = minitorch.TensorBackend(minitorch.CudaOps)
    print("Using GPU backend")
else:
    backend = minitorch.TensorBackend(minitorch.FastOps)
    print("Using CPU backend")

model = Network(backend=backend)
```

### Saving and Loading Models

```python
# Save model weights
model.save_weights("model.npz")

# Load model weights
model.load_weights("model.npz")
```

### Available Optimizers

- `SGD(parameters, lr, momentum)` - Stochastic Gradient Descent with optional momentum
- `RMSProp(parameters, lr, decay_rate, eps)` - RMSProp optimizer

### Available Loss Functions

- `nll_loss(output, target)` - Negative Log Likelihood Loss
- `bce_loss(output, target)` - Binary Cross Entropy Loss
- `cross_entropy_loss(output, target)` - Cross Entropy Loss
- `mse_loss(output, target)` - Mean Squared Error Loss

### Example: Training MNIST

See `examples/run_mnist_multiclass.py` for a complete example of training a LeNet-5 CNN on MNIST:

```bash
python examples/run_mnist_multiclass.py --backend gpu --batch_size 32 --epochs 10 --lr 0.01
```

## TODO
- ~~Implement vanilla RNN layer~~
- Implement Adam optimizer