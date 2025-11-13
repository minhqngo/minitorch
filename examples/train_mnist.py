import argparse
import numba
import numpy as np
import os
import shutil
import sys
import warnings
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter
from tqdm import tqdm

import minitorch
from minitorch.datasets import mnist
from minitorch.dataloader import DataLoader

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

H, W = 28, 28
C = 10


def preprocess(image):
    return image.astype(np.float64) / 255.0


class LeNetOriginal(minitorch.Module):  # LeNet-5
    def __init__(self, backend=FastTensorBackend):
        super().__init__()
        self.conv1 = minitorch.Conv2d(in_channels=1, out_channels=6, kernel=(5, 5), stride=1, backend=backend)
        self.conv2 = minitorch.Conv2d(in_channels=6, out_channels=16, kernel=(5, 5), stride=1, backend=backend)

        self.fc1 = minitorch.Linear(16 * 4 * 4, 120, backend=backend)
        self.fc2 = minitorch.Linear(120, 84, backend=backend)
        self.fc3 = minitorch.Linear(84, C, backend=backend)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x).sigmoid()
        x = minitorch.avgpool2d(x, kernel=(2, 2), stride=(2, 2))
        x = self.conv2(x).sigmoid()
        x = minitorch.avgpool2d(x, kernel=(2, 2), stride=(2, 2))
        x = x.view(batch_size, 16 * 4 * 4)
        x = self.fc1(x).sigmoid()
        x = minitorch.dropout(x, 0.2, not self.training)
        x = self.fc2(x).sigmoid()
        x = minitorch.dropout(x, 0.2, not self.training)
        x = self.fc3(x)
        x = minitorch.logsoftmax(x, dim=1)
        return x


class ModernLeNet(minitorch.Module):
    def __init__(self, backend=FastTensorBackend):
        super().__init__()
        self.conv1 = minitorch.Conv2d(in_channels=1, out_channels=8, kernel=(3, 3), stride=1, backend=backend)
        self.conv2 = minitorch.Conv2d(in_channels=8, out_channels=16, kernel=(3, 3), stride=1, backend=backend)
        self.fc = minitorch.Linear(16 * 5 * 5, C, backend=backend)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x).relu()
        x = minitorch.maxpool2d(x, kernel=(2, 2), stride=(2, 2))
        x = self.conv2(x).relu()
        x = minitorch.maxpool2d(x, kernel=(2, 2), stride=(2, 2))
        x = x.view(batch_size, 16 * 5 * 5)
        x = self.fc(x)
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
    logger=None,
    learning_rate=1e-2,
    max_epochs=50,
    log_fn=default_log_fn,
):  
    optim = minitorch.RMSProp(model.parameters(), learning_rate)
    best_val_acc = float('-inf')
    for epoch in range(1, max_epochs + 1):
        total_loss = 0.0
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train epoch {epoch}/{max_epochs}")
        for i, (X_train, y_train) in pbar:
            optim.zero_grad()
            out = model.forward(X_train.view(X_train.shape[0], 1, H, W))
            loss = minitorch.nll_loss(out, y_train)
            loss.backward()

            total_loss += loss.item()
            optim.step()
            pbar.set_postfix(loss=loss.item())
            
            if logger:
                logger.add_scalar('Loss/train', loss.item(), (epoch - 1) * len(train_loader) + (i + 1))

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
            
        if best_val_acc < correct / total:
            best_val_acc = correct / total
            model.save_weights("mnist_model.npz")
            print("Model saved to mnist_model.npz")
                
        logger.add_scalar('Accuracy/val', correct / total * 100, epoch)
        log_fn(epoch, total_loss, correct, total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", help="backend mode")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="/home/minh/datasets/", help="Directory containing MNIST dataset")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory to log training parameters")
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
        transform=preprocess
    )
    val_loader = DataLoader(
        mnist_val,
        batch_size=args.batch_size,
        shuffle=False,
        backend=backend,
        transform=preprocess
    )

    model = ModernLeNet(backend=backend)
    
    logger = None
    if args.log_dir:
        if os.path.exists(args.log_dir):
            shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)
        logger = SummaryWriter(args.log_dir)

    print("Starting training...")
    train(model, train_loader, val_loader, logger, args.lr, max_epochs=args.epochs)
