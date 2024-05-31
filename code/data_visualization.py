import json
import os
import random

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

matplotlib.use('Agg')


def plot_class_distribution(dataset_dict):
    """
    Plots the class distribution of the dataset in a bar graph.
    :param dataset_dict: the dictionary that contains the file paths and labels
    :return:
    """
    labels = [value['label'] for value in dataset_dict.values()]
    sns.countplot(x=labels)
    plt.title('Class Distribution of Facial Expression Images')
    plt.ylabel('Number of Images')
    plt.xlabel('Class')
    plt.savefig(f'{generated_plots_dir}/class_distribution.png')
    plt.close()


def plot_pixel_intensity_distribution(dataset_dict, labels, sample_size=100):
    """
    Plots the pixel intensity distribution for each class.
    Only a subset of images is processed for efficiency.
    :param sample_size: size o the sample to be used for the visualization
    :param labels: classes to visualize
    :param dataset_dict: the dictionary that contains the file paths and labels
    """

    for label in labels:
        all_pixels = []
        class_images = [key for key, value in dataset_dict.items() if value['label'] == label]

        # only applies the visualization to a subset of the images to save considerable time
        if len(class_images) > sample_size:
            class_images = random.sample(class_images, sample_size)

        for img_key in class_images:
            image_path = dataset_dict[img_key]['normalized_path']
            with Image.open(image_path) as image:
                all_pixels.extend(np.array(image).flatten())

        sns.histplot(all_pixels, kde=True)
        plt.title(f'Pixel Intensity Distribution for class: {label}')
        plt.xlabel('Pixel Intensity (0-255)')
        plt.ylabel('Frequency')
        plt.savefig(f'{generated_plots_dir}/pixel_intensity_distribution_{label}.png')
        plt.close()


def get_dataset_dictionary():
    """
    This function reads the dataset.json file and returns the dictionary
    :return:
    """
    with open('dataset.json', 'r') as json_file:
        return json.load(json_file)


def get_paths_from_label(data_dict, label, path_type='original_path', num_paths=15):
    """
    Helper for extracting an arbitrary number of paths from a specific label in the dataset dictionary.

    Parameters:
    - data_dict: The dictionary containing paths and labels for all images
    - label: the class of interest
    - path_type: 'original_path' or 'normalized_path'
    - num_paths: number of paths to extract

    Returns:
    - A list of paths for the needed class containing exactly num_paths paths
    """
    # get only the paths for the specified label
    label_paths = {key: value for key, value in data_dict.items() if value['label'] == label}

    # get the needed path type
    original_images_paths = [image_data[path_type] for image_data in label_paths.values()]

    # select randomly the paths to return
    selected_paths = random.sample(original_images_paths, num_paths)

    return selected_paths


def plot_images_with_histograms(image_paths, class_name):
    """
    This function plots the images and their pixel intensity histograms for a given class.
    :param image_paths: list containing the paths of the images to display
    :param class_name: name of the class that is being visualized
    :return:
    """
    fig, axs = plt.subplots(5, 6, figsize=(20, 15))
    axs = axs.ravel()  # flatten the array to access it with 1 index

    for i, img_path in enumerate(image_paths):
        # plot the image
        img = cv2.imread(img_path)
        axs[2 * i].imshow(img)
        axs[2 * i].axis('off')

        # plot the distribution of pixels histogram
        axs[2 * i + 1].hist(img.ravel(), bins=256, range=[0, 256])
        axs[2 * i + 1].set_title('Histogram')
        axs[2 * i + 1].set_xlabel('Pixel Intensity')
        axs[2 * i + 1].set_ylabel('Frequency')

    fig.suptitle(f'Sample images and their pixel distribution histograms for class: {class_name}')
    plt.tight_layout()
    plt.savefig(f'{generated_plots_dir}/{class_name}_sample_images_with_histograms.png')
    plt.close()


generated_plots_dir = "plots"
os.makedirs(generated_plots_dir, exist_ok=True)

# load the dataset dictionary
dataset_dict = get_dataset_dictionary()

# Class names
classes = ['angry', 'happy', 'neutral', "focused"]

# visualization
for class_name in classes:
    paths = get_paths_from_label(dataset_dict, class_name)
    # plot the images and histograms
    plot_images_with_histograms(paths, class_name)

plot_class_distribution(dataset_dict)
plot_pixel_intensity_distribution(dataset_dict, classes)
