import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from cnn_model import SimpleCNN

transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = datasets.ImageFolder('data/normalized_data', transform=transform)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = []

# apply kfold cross validation
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    # datasets
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)

    train_size = int(0.85 * len(train_subset))
    val_size = len(train_subset) - train_size
    train_data, val_data = random_split(train_subset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    # model, loss function and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    for epoch in range(20):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            break

    # evaluate on test set
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    results.append({
        'fold': fold + 1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

# calculate average results to see the 10-fold performance
avg_accuracy = sum([result['accuracy'] for result in results]) / 10
avg_precision = sum([result['precision'] for result in results]) / 10
avg_recall = sum([result['recall'] for result in results]) / 10
avg_f1 = sum([result['f1_score'] for result in results]) / 10

print("10-Fold Cross-Validation Results")
for result in results:
    print(result)
print(f"Average Accuracy: {avg_accuracy}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1 Score: {avg_f1}")
