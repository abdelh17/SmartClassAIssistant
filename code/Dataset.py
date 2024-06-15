import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

torch.manual_seed(20)


def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


device = get_device()


class DatasetLoader:
    def __init__(self, data_dir='data/normalized_data', batch_size=64):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.train_size = int(0.7 * len(self.dataset))
        self.val_size = int(0.15 * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [self.train_size, self.val_size, self.test_size])
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def get_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def get_class_names(self):
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
