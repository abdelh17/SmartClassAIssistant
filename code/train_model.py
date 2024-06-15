import os

import torch
import torch.nn as nn

from Dataset import DatasetLoader, calculate_accuracy, get_device
from cnn_model import SimpleCNN, Model2, Model3

# set random seed for reproducibility
torch.manual_seed(20)

# for gpu use to speed up training
device = get_device()
print("Using device: ", device)

# data loaders
dataset = DatasetLoader(data_dir='data/normalized_data', batch_size=64)
train_loader, val_loader, test_loader = dataset.get_loaders()

# create a directory to save the best models
os.makedirs('saved_models', exist_ok=True)

models = [SimpleCNN(), Model2(), Model3()]

# training loop
for model in models:
    print("Training model: ", model.__class__.__name__)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    patience = 12
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_accuracy = calculate_accuracy(train_loader, model)

        model.eval()
        val_loss = 0.0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        val_accuracy = calculate_accuracy(val_loader, model)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, '
              f'Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')

        # save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            epoch_loaded = epoch + 1
            torch.save(model.state_dict(), f'saved_models/best_model_{model.__class__.__name__}.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping, stopped at epoch: ", epoch + 1)
            break
    print("Epoch loaded: ", epoch_loaded, "for model: ", model.__class__.__name__)
