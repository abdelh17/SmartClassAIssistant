import os
from PIL import Image
from torch.utils.data import Dataset

class FacialExpressionDataset(Dataset):
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
