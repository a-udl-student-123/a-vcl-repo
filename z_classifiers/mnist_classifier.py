# mnist classifier for evaluating generated samples
# implements cnn classifier to check quality of generated digits

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

MODEL_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / 'data'


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3 and x.size(1)==28 and x.size(2)==28:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train_mnist_classifier(model, device, save_path, batch_size=64, epochs=100, lr=0.01, patience=10):
    """trains classifier from scratch with early stopping
    
    Args:
        model: classifier to train
        device: cuda/cpu
        save_path: where to save model
        batch_size: batch size for training
        epochs: max epochs
        lr: learning rate
        patience: early stop after N epochs w/no improvement
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                      ])),
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                      ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    model.train()
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0  

    # TODO

    progress_bar = tqdm(range(epochs), desc="Training MNIST classifier")
    for epoch in progress_bar:
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        correct = 0
        total = 0
        running_loss=0.0
        
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            running_loss+=loss.item()
            
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * correct / total:.1f}%"
            })
            
        train_acc = 100. * correct / total

        # eval on validation set
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        val_acc = 100. * correct / total
        val_loss = val_loss / total if total > 0 else 0.0

        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        scheduler.step(avg_loss)

        progress_bar.set_postfix({
            "train": f"{train_acc:.1f}%",
            "val": f"{val_acc:.1f}%", 
            "best": f"{best_val_acc:.1f}%"
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            progress_bar.set_description(f"Early stopping: epoch {epoch+1}/{epochs}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    torch.save(model.state_dict(), save_path)
    
    return model, best_val_acc


def evaluate_classifier(model, device, batch_size=1000):
    # evaluates accuracy on test set
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                      ])),
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy


def get_mnist_classifier(device, force_train=False):
    # loads pretrained classifier or trains new one
    model_path = MODEL_DIR / 'mnist_classifier.pt'
    
    model = MNISTClassifier().to(device)
    
    if model_path.exists() and not force_train:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            tqdm.write("Loaded pre-trained MNIST classifier")
        except Exception as e:
            tqdm.write(f"Error loading classifier: {str(e)}")
            model, _ = train_mnist_classifier(model, device, model_path, patience=5)
    else:
        model, _ = train_mnist_classifier(model, device, model_path, patience=5)

    model.eval()
    return model