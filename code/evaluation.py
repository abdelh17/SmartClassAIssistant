import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from Dataset import DatasetLoader, get_device
from cnn_model import SimpleCNN, Model3, Model2

matplotlib.use('Agg')


def generate_confusion_matrix(cm, model_name):
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'confusion_matrix/confusion_matrix_{model_name}.png')
    plt.close()


def evaluate_metrics(all_preds, all_labels, test_loss, model_name):
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Testing model: {model_name}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro',
                                                                                 zero_division=0)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro',
                                                                                 zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.2f}")
    print(f'MACRO: Precision: {macro_precision:.2f}, Recall: {macro_recall:.2f}, F1-score: {macro_f1:.2f}')
    print(f'MICRO: Precision: {micro_precision:.2f}, Recall: {micro_recall:.2f}, F1-score: {micro_f1:.2f}')
    print()

    return cm


device = get_device()

dataset = DatasetLoader(data_dir='data/normalized_data', batch_size=64)
_, test_loader, _ = dataset.get_loaders()
class_labels = dataset.get_class_names()

# create a directory to save the confusion matrices
os.makedirs('confusion_matrix', exist_ok=True)

models = [SimpleCNN(), Model2(), Model3()]

# Load and test each model
for model in models:
    model_name = model.__class__.__name__
    model_path = f'saved_models/best_model_{model_name}.pth'
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    test_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        test_loss /= len(test_loader)

    cm = evaluate_metrics(all_preds, all_labels, test_loss, model_name)

    generate_confusion_matrix(cm, model_name)
