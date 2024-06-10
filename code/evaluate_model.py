import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from cnn_model import SimpleCNN, Variant1CNN, Variant2CNN
from facial_expression_dataset import FacialExpressionDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
import os

def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, precision, recall, f1, cm


img_dir = os.path.join(os.getcwd(), 'data', 'normalized_data')

if not os.path.exists(img_dir):
    raise FileNotFoundError(f"Image directory not found at {img_dir}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = FacialExpressionDataset(img_dir=img_dir, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

_, _, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Evaluate the main model
main_model = SimpleCNN()
main_model.load_state_dict(torch.load('best_model.pth'))
accuracy, precision, recall, f1, cm = evaluate_model(main_model, test_loader)
print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}')

matplotlib.use('Agg')

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', xticklabels=dataset.class_names, yticklabels=dataset.class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
