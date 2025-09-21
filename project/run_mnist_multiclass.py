import argparse
import numba
import sys

import minitorch
from minitorch.datasets import mnist
from minitorch.dataloader import DataLoader

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


class Network(minitorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.
    This model should implement the following procedure:
    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self, backend=FastTensorBackend):
        super().__init__()

        # For vis
        self.mid = None
        self.out = None

        self.conv1 = minitorch.Conv2d(1, 4, 3, 3, backend=backend)
        self.conv2 = minitorch.Conv2d(4, 8, 3, 3, backend=backend)
        self.linear1 = minitorch.Linear(392, 64, backend=backend)
        self.linear2 = minitorch.Linear(64, C, backend=backend)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x).relu()
        self.mid = x
        x = self.conv2(x).relu()
        self.out = x
        x = minitorch.avgpool2d(x, (4, 4))
        x = x.view(batch_size, 392)
        x = self.linear1(x).relu()
        x = minitorch.dropout(x, 0.25, self.mode == "eval")
        x = self.linear2(x)
        x = minitorch.logsoftmax(x, dim=1)
        return x


def default_log_fn(epoch, total_loss, correct, total, model):
    print(
        f"Epoch {epoch} | loss {total_loss / total:.2f} | valid acc {correct / total:.2f}"
    )


def train(
    model,
    train_loader,
    val_loader,
    learning_rate,
    max_epochs=50,
    log_fn=default_log_fn,
    backend=None,
):
    optim = minitorch.SGD(model.parameters(), learning_rate)
    for epoch in range(1, max_epochs + 1):
        total_loss = 0.0
        model.train()
        for X_train, y_train in train_loader:
            # Forward
            out = model.forward(X_train.view(X_train.shape[0], 1, H, W))
            
            # Compute NLL loss
            loss = minitorch.nll_loss(out, y_train)
            loss.backward()

            total_loss += loss.item()

            # Update
            optim.step()

        # Evaluate on validation set
        correct = 0
        total = 0
        model.eval()
        for X_val, y_val in val_loader:            
            out = model.forward(X_val.view(X_val.shape[0], 1, H, W))
            
            # Get predictions
            y_hat = out.argmax(dim=1)

            correct += (y_hat == y_val).sum().item()
            total += y_val.shape[0]

        log_fn(epoch, total_loss, correct, total, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", help="backend mode")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    if args.backend == "gpu" and numba.cuda.is_available():
        backend = GPUBackend
        print("Using CUDA backend")
    else:
        if args.backend == "gpu":
            print("CUDA backend not available, using CPU instead.", file=sys.stderr)
        backend = FastTensorBackend
        print("Using CPU backend")

    # Load MNIST data
    mnist_train = mnist.MNISTDataset("/home/minh/datasets/", train=True)
    mnist_val = mnist.MNISTDataset("/home/minh/datasets/", train=False)

    # Create data loaders
    train_loader = DataLoader(
        mnist_train,
        batch_size=args.batch_size,
        shuffle=True,
        backend=backend
    )
    val_loader = DataLoader(
        mnist_val,
        batch_size=args.batch_size,
        shuffle=False,
        backend=backend
    )

    # Initialize model
    model = Network(backend=backend)

    print("Starting training...")
    train(model, train_loader, val_loader, args.lr, max_epochs=args.epochs, backend=backend)

    # Save model
    model.save_weights("mnist_model.pt")
    print("Model saved to mnist_model.pt")
