import argparse
import numba
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

import minitorch
from minitorch.datasets import mnist
from minitorch.dataloader import DataLoader

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

C = 10

H, W = 28, 28


def mnist_transform(image):
    """Normalize MNIST image from uint8 [0, 255] to float [0, 1]"""
    return image.astype(np.float64) / 255.0


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
        x = minitorch.dropout(x, 0.25, not self.training)
        x = self.linear2(x)
        x = minitorch.logsoftmax(x, dim=1)
        return x


def default_log_fn(epoch, total_loss, correct, total):
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
):
    optim = minitorch.SGD(model.parameters(), learning_rate)
    for epoch in range(1, max_epochs + 1):
        total_loss = 0.0
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Train epoch {epoch}/{max_epochs}")
        for X_train, y_train in pbar:
            optim.zero_grad()
            out = model.forward(X_train.view(X_train.shape[0], 1, H, W))
            loss = minitorch.nll_loss(out, y_train)
            loss.backward()

            total_loss += loss.item()
            optim.step()
            pbar.set_postfix(loss=loss.item())

        correct = 0
        total = 0
        model.eval()
        pbar = tqdm(val_loader, total=len(val_loader), desc=f"Val epoch {epoch}/{max_epochs}")
        for X_val, y_val in pbar:            
            out = model.forward(X_val.view(X_val.shape[0], 1, H, W))
            y_hat = minitorch.argmax(out, dim=1).squeeze()
            correct += (y_hat == y_val).sum().item()
            total += y_val.shape[0]
            pbar.set_postfix(acc=correct / total * 100)

        log_fn(epoch, total_loss, correct, total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", help="backend mode")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="/home/minh/datasets/", help="Directory containing MNIST dataset")
    args = parser.parse_args()

    if args.backend == "gpu" and numba.cuda.is_available():
        backend = GPUBackend
        print("Using CUDA backend")
    else:
        if args.backend == "gpu":
            print("CUDA backend not available, using CPU instead.", file=sys.stderr)
        backend = FastTensorBackend
        print("Using CPU backend")

    mnist_train = mnist.MNISTDataset(args.data_dir, train=True)
    mnist_val = mnist.MNISTDataset(args.data_dir, train=False)

    train_loader = DataLoader(
        mnist_train,
        batch_size=args.batch_size,
        shuffle=True,
        backend=backend,
        transform=mnist_transform
    )
    val_loader = DataLoader(
        mnist_val,
        batch_size=args.batch_size,
        shuffle=False,
        backend=backend,
        transform=mnist_transform
    )

    model = Network(backend=backend)

    print("Starting training...")
    train(model, train_loader, val_loader, args.lr, max_epochs=args.epochs)

    model.save_weights("mnist_model.npz")
    print("Model saved to mnist_model.npz")
