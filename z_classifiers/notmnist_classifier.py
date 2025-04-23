# notMNIST classifier for evaluating generated samples
# cnn classifier to check quality of generated letters

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from z_data.datasets import notMNISTDataset

MODEL_DIR = Path(__file__).parent
DATA_DIR = 'data'


class NotMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # same arch as mnist classifier
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)  # A-J classes

    def forward(self, x):
        # handle diff input formats
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3 and x.size(1)==28 and x.size(2) == 28:
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


def train_notmnist_classifier(model, device, save_path, batch_size=64, epochs=20, lr=0.01, patience=5):
    """trains notMNIST classifier with early stopping and saves best model"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Using lambda x:x since dataset already returns tensors
    train_dataset = notMNISTDataset(DATA_DIR, train=True, download=True, transform=lambda x: x)
    test_dataset = notMNISTDataset(DATA_DIR, train=False, download=True, transform=lambda x: x)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)  # 0 workers for windows
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0


    model.train()
    progress_bar = tqdm(range(epochs), desc="Training notMNIST classifier")
    for epoch in progress_bar:
        # train phase
        model.train()
        running_loss=0.0
        correct = 0
        total = 0
        
        batch_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for data, target in batch_progress_bar:
            data, target=data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # track acc
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            running_loss += loss.item()

            batch_progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct/total:.1f}%"
            })

        train_acc = 100. * correct / total if total > 0 else 0.0

        # validation 
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)

        val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_loss / val_total if val_total > 0 else 0.0

        progress_bar.set_postfix({
            'train': f"{train_acc:.1f}%",
            'val': f"{val_acc:.1f}%", 
            'best': f"{best_val_acc:.1f}%"
        })

        # update scheduler
        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        scheduler.step(avg_loss)

        # track best model
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
    test_dataset = notMNISTDataset(DATA_DIR, train=False, download=True, transform=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)

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

    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy


def get_notmnist_classifier(device, force_train=False):
    # loads pretrained classifier or trains new one
    model_path = MODEL_DIR / 'notmnist_classifier.pt'
    model = NotMNISTClassifier().to(device)

    if model_path.exists() and not force_train:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            tqdm.write("Loaded pre-trained notMNIST classifier")
        except Exception as e:
            tqdm.write(f"Error loading classifier: {str(e)}")
            model, _ = train_notmnist_classifier(model, device, model_path)
    else:
        model, _ = train_notmnist_classifier(model, device, model_path)

    model.eval()
    return model