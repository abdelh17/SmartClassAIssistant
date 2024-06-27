import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from Dataset import get_device
from cnn_model import SimpleCNN

# set random seed for reproducibility
torch.manual_seed(20)

transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = datasets.ImageFolder('data/normalized_data', transform=transform)
device = get_device()
kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = []

# apply kfold cross validation
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Starting fold {fold + 1}...")
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
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    patience = 12
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Fold {fold + 1}: Early stopping, stopped at epoch: ", epoch + 1)
            break

    # evaluate on test set
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro',
                                                                                 zero_division=1)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro',
                                                                                 zero_division=1)

    results.append({
        'fold': fold + 1,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1_score': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1_score': micro_f1
    })

# calculate average results to see the 10-fold performance
avg_accuracy = sum([result['accuracy'] for result in results]) / 10
avg_macro_precision = sum([result['macro_precision'] for result in results]) / 10
avg_macro_recall = sum([result['macro_recall'] for result in results]) / 10
avg_macro_f1 = sum([result['macro_f1_score'] for result in results]) / 10

avg_micro_precision = sum([result['micro_precision'] for result in results]) / 10
avg_micro_recall = sum([result['micro_recall'] for result in results]) / 10
avg_micro_f1 = sum([result['micro_f1_score'] for result in results]) / 10

print("\n10-Fold Cross-Validation Results")
print("---------------------------------------------")
for result in results:
    print(f"Fold {result['fold']}: "
          f"Accuracy: {result['accuracy']:.2f}, "
          f"Macro Precision: {result['macro_precision']:.2f}, "
          f"Macro Recall: {result['macro_recall']:.2f}, "
          f"Macro F1 Score: {result['macro_f1_score']:.2f}, "
          f"Micro Precision: {result['micro_precision']:.2f}, "
          f"Micro Recall: {result['micro_recall']:.2f}, "
          f"Micro F1 Score: {result['micro_f1_score']:.2f}")

print("\nAverage Results")
print("---------------------------------------------")
print(f"Average Accuracy: {avg_accuracy:.2f}")
print(f"Average Macro Precision: {avg_macro_precision:.2f}")
print(f"Average Macro Recall: {avg_macro_recall:.2f}")
print(f"Average Macro F1 Score: {avg_macro_f1:.2f}")

print(f"Average Micro Precision: {avg_micro_precision:.2f}")
print(f"Average Micro Recall: {avg_micro_recall:.2f}")
print(f"Average Micro F1 Score: {avg_micro_f1:.2f}")
