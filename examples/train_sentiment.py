"""Train a 1D CNN on sentiment classification data"""

import argparse
import embeddings
import numba
import numpy as np
import os
import shutil
import sys
import warnings
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import minitorch
from minitorch.datasets import uci_sentiment
from minitorch.dataloader import DataLoader

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)
    
    
class CNNSentiment(minitorch.Module):
    def __init__(self, feature_map_size=100, embedding_size=50, filter_sizes=[3, 4, 5], backend=FastTensorBackend):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.conv1 = minitorch.Conv1d(embedding_size, feature_map_size, filter_sizes[0], backend=backend)
        self.conv2 = minitorch.Conv1d(embedding_size, feature_map_size, filter_sizes[1], backend=backend)
        self.conv3 = minitorch.Conv1d(embedding_size, feature_map_size, filter_sizes[2], backend=backend)
        self.fc = minitorch.Linear(feature_map_size, 1, backend=backend)
        
    def forward(self, embeddings):
        x = embeddings.permute(0, 2, 1)
        x1 = self.conv1(x).relu()
        x2 = self.conv2(x).relu()
        x3 = self.conv3(x).relu()
        x = minitorch.max(x1, 2) + minitorch.max(x2, 2) + minitorch.max(x3, 2)
        x = self.fc(x.view(x.shape[0], self.feature_map_size))
        x = minitorch.dropout(x, 0.2, not self.training)
        return x.sigmoid().view(x.shape[0])
    
    
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
            out = model.forward(X_train)
            prob = (out * y_train) + (out - 1.0) * (y_train - 1.0)
            loss = -(prob.log() / y_train.shape[0]).sum().view(1)
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
            out = model.forward(X_val)
            preds = (out > 0.5).astype(y_val.dtype)
            correct += (preds == y_val).sum().item()
            total += y_val.shape[0]
            pbar.set_postfix(acc=correct / total * 100)
            
        if best_val_acc < correct / total:
            best_val_acc = correct / total
            model.save_weights("sentiment_model.npz")
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
        
    emb_lookup = embeddings.GloveEmbedding("wikipedia_gigaword", d_emb=50, show_progress=True)
    ds = uci_sentiment.UCISentimentDataset(root=args.data_dir, emb_lookup=emb_lookup)
    sentiment_train, sentiment_val = train_test_split(ds, test_size=0.2, random_state=42)
    train_loader = DataLoader(
        sentiment_train,
        batch_size=args.batch_size,
        shuffle=True,
        backend=backend
    )
    val_loader = DataLoader(
        sentiment_val,
        batch_size=args.batch_size,
        shuffle=False,
        backend=backend
    )
    
    model = CNNSentiment(feature_map_size=100, embedding_size=50, filter_sizes=[3, 4, 5], backend=backend)
    
    logger = None
    if args.log_dir:
        if os.path.exists(args.log_dir):
            shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)
        logger = SummaryWriter(args.log_dir)

    print("Starting training...")
    train(model, train_loader, val_loader, logger, args.lr, max_epochs=args.epochs)
