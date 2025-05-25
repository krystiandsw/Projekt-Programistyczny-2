import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SimpleNN
from train import train_model
from test import evaluate_model, show_predictions
from utils import save_model, load_model

# Przygotowanie danych
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Inicjalizacja modelu
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening
train_model(model, train_loader, criterion, optimizer, epochs=5)

# Zapis modelu
save_model(model)

# Ewaluacja
evaluate_model(model, test_loader)
show_predictions(model, test_loader)