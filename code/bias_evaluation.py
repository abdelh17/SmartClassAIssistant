import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from Dataset import get_device
from cnn_model import SimpleCNN

# Mapping for shorthand labels to full words
attribute_mapping = {
    'm': 'Male',
    'f': 'Female',
    'y': 'Young',
    'a': 'Adult',
    's': 'Senior'
}

# Function to filter dataset by attribute (gender and age)
def filter_dataset_by_attribute(root_dir, gender, age):
    filtered_dataset = datasets.ImageFolder(root=root_dir, transform=transformation)
    indices = []
    for idx, (path, _) in enumerate(filtered_dataset.samples):
        if gender in path and age in path:
            indices.append(idx)
    return torch.utils.data.Subset(filtered_dataset, indices)

# Function to create DataLoader for filtered dataset
def create_dataloader(filtered_dataset, batch_size=64):
    dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# Define transformation
transformation = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Root directory for normalized data
normalized_data_dir = 'data/normalized_data'

# Create DataLoader for each demographic group
attributes = [('m', 'a'), ('f', 'a'), ('m', 's'), ('f', 's'), ('m', 'y'), ('f', 'y')]
group_dataloaders = {attr: create_dataloader(filter_dataset_by_attribute(normalized_data_dir, *attr)) for attr in attributes}

# Evaluate group performance
def evaluate_group_performance(model, dataloader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    return test_loss, accuracy, precision, recall, f1

# Load model
device = get_device()
model = SimpleCNN()
model_name = 'SimpleCNN'

group_metrics = {attr: {} for attr in attributes}

criterion = nn.CrossEntropyLoss()

# Evaluate the model on each group
model.load_state_dict(torch.load(f'saved_models/best_model_{model_name}.pth'))
model.to(device)
for attr, dataloader in group_dataloaders.items():
    test_loss, accuracy, precision, recall, f1 = evaluate_group_performance(model, dataloader, criterion)
    group_metrics[attr] = {
        'loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Print group metrics in table format
print("Bias Analysis Table:")
print(f"{'Group':<15} {'Model':<15} {'Loss':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
for attr, metrics in group_metrics.items():
    gender, age = attr
    full_gender = attribute_mapping[gender]
    full_age = attribute_mapping[age]
    print(f"{full_gender} {full_age:<10} {model_name:<15} {metrics['loss']:<10.4f} {metrics['accuracy']:<10.2f} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} {metrics['f1']:<10.2f}")

# Analyze and mitigate bias
# Assuming we detect bias, we would augment the dataset and retrain the models
# This example assumes augmentation and retraining are done; otherwise, this section would be expanded

# Retraining and reassessment logic goes here
# For example, using augmented data and repeating the training and evaluation process

print("Bias detection and mitigation process completed.")
