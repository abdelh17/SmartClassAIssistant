import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
from Dataset import get_device, calculate_accuracy
from cnn_model import SimpleCNN

# Mapping for shorthand labels to full words
attribute_mapping = {
    'm': 'Male',
    'f': 'Female',
    'y': 'Young',
    'a': 'Adult',
    's': 'Senior'
}

# Define augmentation transformation
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define normal transformation
transformation = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to filter dataset by attribute (gender and age)
def filter_dataset_by_attribute(root_dir, gender, age, transform):
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    indices = [idx for idx, (path, _) in enumerate(dataset.samples) if gender in path and age in path]
    return torch.utils.data.Subset(dataset, indices)

# Function to create DataLoader for filtered dataset
def create_dataloader(filtered_dataset, batch_size=64):
    return DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)

# Function to augment and combine datasets for each demographic group
def augment_and_combine_datasets(normalized_data_dir):
    combined_datasets = []
    attributes_to_augment = [('m', 'a'), ('f', 'a'), ('m', 's'), ('f', 's'), ('m', 'y'), ('f', 'y')]

    for gender, age in attributes_to_augment:
        full_gender = attribute_mapping[gender]
        full_age = attribute_mapping[age]
        print(f"Augmenting dataset for group {full_gender} {full_age}...")

        original_dataset = filter_dataset_by_attribute(normalized_data_dir, gender, age, transformation)
        augmented_dataset = filter_dataset_by_attribute(normalized_data_dir, gender, age, augmentation_transforms)

        combined_dataset = ConcatDataset([original_dataset, augmented_dataset])
        combined_datasets.append(combined_dataset)

    return ConcatDataset(combined_datasets)

# Root directory for normalized data
normalized_data_dir = 'data/normalized_data'

# Load model
device = get_device()
model = SimpleCNN()
model_name = 'SimpleCNN'
model.to(device)

# Create a directory to save the best models
os.makedirs('saved_models', exist_ok=True)

# Training settings
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
patience = 12

# Create DataLoader for combined dataset
print("Starting dataset augmentation and model retraining...")
combined_dataset = augment_and_combine_datasets(normalized_data_dir)
combined_dataloader = create_dataloader(combined_dataset)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in combined_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_accuracy = calculate_accuracy(combined_dataloader, model)

    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for images, labels in combined_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        val_accuracy = calculate_accuracy(combined_dataloader, model)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(combined_dataloader):.4f}, '
          f'Val Loss: {val_loss / len(combined_dataloader):.4f}, '
          f'Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')

    # Save the model with the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), f'saved_models/best_model_{model_name}.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping, stopped at epoch: ", epoch + 1)
        break

print("Augmentation and retraining process completed.")
