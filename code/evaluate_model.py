import os
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from cnn_model import SimpleCNN, Variant1CNN, Variant2CNN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class PreprocessedDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        self.class_names = os.listdir(img_dir)
        for label in self.class_names:
            label_dir = os.path.join(img_dir, label)
            for img_name in os.listdir(label_dir):
                self.img_labels.append((os.path.join(label_dir, img_name), label))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_names.index(label)
        return image, label_idx

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

# Path to the directory containing normalized data
img_dir = os.path.join(os.getcwd(), 'data', 'normalized_data')

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = PreprocessedDataset(img_dir=img_dir, transform=transform)

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

# Explicitly set the backend to Agg
import matplotlib
matplotlib.use('Agg')

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', xticklabels=dataset.class_names, yticklabels=dataset.class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save the plot as an image file
plt.close()  # Close the plot to avoid any backend issues
