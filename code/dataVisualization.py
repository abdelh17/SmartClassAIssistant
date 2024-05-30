import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

matplotlib.use('Agg')


def create_dataset_dict(base_dir):
    """
    Create a dataset dictionary from the normalized data directory.
    """
    dataset_dict = {}
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith('.jpg'):
                    identifier = filename.split('.')[0]
                    dataset_dict[identifier] = {
                        "normalized_path": os.path.join(label_dir, filename),
                        "label": label
                    }

    return dataset_dict


def plot_class_distribution(dataset_dict):
    """
    Plots the class distribution.
    """
    labels = [value['label'] for value in dataset_dict.values()]
    sns.countplot(x=labels)
    plt.title('Class Distribution')
    plt.savefig('data/class_distribution.png')  # Save the plot as a PNG file
    plt.close()


def plot_pixel_intensity_distribution(dataset_dict, sample_size=100):
    """
    Plots the pixel intensity distribution for each class.
    Only a subset of images is processed for efficiency.
    """
    for label in set([value['label'] for value in dataset_dict.values()]):
        all_pixels = []
        class_images = [key for key, value in dataset_dict.items() if value['label'] == label]

        # If there are more images than sample_size, randomly sample them
        if len(class_images) > sample_size:
            class_images = random.sample(class_images, sample_size)

        for img_key in class_images:
            image_path = dataset_dict[img_key]['normalized_path']
            with Image.open(image_path) as image:
                all_pixels.extend(np.array(image).flatten())

        sns.histplot(all_pixels, kde=True)
        plt.title(f'Pixel Intensity Distribution for {label}')
        plt.savefig(f'data/pixel_intensity_distribution_{label}.png')  # Save the plot as a PNG file
        plt.close()


def plot_sample_images(dataset_dict, samples_per_class=5):
    """
    Displays a grid of sample images from each class.
    """
    classes = list(set([value['label'] for value in dataset_dict.values()]))
    num_classes = len(classes)

    # Calculate the grid size
    cols = samples_per_class
    rows = num_classes

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle('Sample Images from Each Class')

    for i, label in enumerate(classes):
        class_images = [key for key, value in dataset_dict.items() if value['label'] == label]
        selected_images = random.sample(class_images, min(samples_per_class, len(class_images)))
        for j, img_key in enumerate(selected_images):
            img_path = dataset_dict[img_key]['normalized_path']
            image = Image.open(img_path)
            ax = axs[i, j]
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(label)

    # Hide any remaining empty subplots
    for i in range(rows):
        for j in range(len(selected_images), cols):
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('data/sample_images.png')  # Save the plot as a PNG file
    plt.close()


# Main code for visualization
base_dir = 'data/normalized_data'

# Create the dataset dictionary
dataset_dict = create_dataset_dict(base_dir)

# Dataset visualization
plot_class_distribution(dataset_dict)
plot_pixel_intensity_distribution(dataset_dict)
plot_sample_images(dataset_dict)
