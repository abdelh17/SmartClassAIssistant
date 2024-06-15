import argparse
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

from Dataset import get_device, DatasetLoader
from cnn_model import SimpleCNN, Model3, Model2

output_dir = 'model_outputs'
device = get_device()

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # standard normalization values
])

# load models
models = [SimpleCNN(), Model3(), Model2()]

for model in models:
    model_name = model.__class__.__name__
    model_path = f'saved_models/best_model_{model_name}.pth'
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

dataset = DatasetLoader()


def load_image(path_to_image):
    image = Image.open(path_to_image).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def predict_single_image(path_to_image):
    """
    Predicts the class of a single image
    :param path_to_image: single path to an image
    :return:
    """
    image = load_image(path_to_image).to(device)

    predictions = {}
    with torch.no_grad():
        for model in models:
            model_name = model.__class__.__name__
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]
            predictions[model_name] = predicted_class
    return predictions


def predict_multiple_images(image_paths):
    """
    Predicts the class of multiple images
    :param image_paths: list of paths to the images
    :return:
    """
    predictions = {}
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        predictions[image_name] = predict_single_image(image_path)
    return predictions


class_names = dataset.get_class_names()

# for command line use
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification Inference')
    parser.add_argument('image_paths', nargs='+', help='List of paths to images for inference')
    args = parser.parse_args()

    image_paths = args.image_paths
    predictions = predict_multiple_images(image_paths)
    print(f'Predictions for images: {predictions}')
