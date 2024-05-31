import json
import os
import zipfile

import torchvision.transforms as transforms
from PIL import Image


def extract_zip(zip_path):
    """
    This function extracts the zip file to the data folder
    :param zip_path: path to the zip file
    :return:
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data")
        # important if working on MacOS
        if os.path.exists("data/__MACOSX"):
            import shutil
            shutil.rmtree("data/__MACOSX")


def cleanup_root():
    """
    This function removes the folders and files that were created during previous runs
    """
    folders_to_remove = ['data/angry', 'data/happy', 'data/neutral', 'data/focused', 'data/normalized_data']
    for folder in folders_to_remove:
        if os.path.exists(folder):
            import shutil
            shutil.rmtree(folder)
    if os.path.exists('dataset.json'):
        os.remove('dataset.json')


def rename_files(folder_path, label):
    """
    This function renames the files in the folder to be in the format labelNUMBER.jpg.
    It also updates the dataset_dict dictionary with the new file paths
    :param folder_path: path where the files are located
    :param label: label to be added to the file name
    :return:
    """
    files = os.listdir(folder_path)
    # images 1,2,3 are those of the team members
    start_number = 4
    for i, filename in enumerate(files):
        # skip renaming the images of the team members that already the correct naming pattern
        if label in filename:
            identifier = filename.split(".")[0]
            dataset_dict[identifier] = {"original_path": os.path.join(folder_path, filename),
                                        "normalized_path": f"data/normalized_data/{label}/transformed_{identifier}.jpg",
                                        "label": label}
            continue
        # rename all other images
        if filename.endswith('.jpg'):
            identifier = f"{label}{start_number + i}"
            new_file_name = f"{identifier}.jpg"
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(old_file_path, new_file_path)
            dataset_dict[identifier] = {"original_path": new_file_path,
                                        "normalized_path": f"data/normalized_data/{label}/transformed_{identifier}.jpg",
                                        "label": label}


def make_normalized_directories(folder_names):
    """
    This function creates folders for the normalized images
    :param folder_names: List that contains the names of the folders to be created, names are the labels
    :return:
    """

    # folder that contains subfolders for each class
    output_dir = 'data/normalized_data'
    os.makedirs(output_dir, exist_ok=True)
    # create subfolders for each class
    class_dirs = {label: os.path.join(output_dir, label) for label in folder_names}
    for class_dir in class_dirs.values():
        os.makedirs(class_dir, exist_ok=True)


def generate_transformed_images(path_dictionary, transform):
    """
    This function transforms the original images and saves them in the normalized directory
    :param path_dictionary: dictionary of each image that contains the path to the original image and that of the transformed image
    :param transform: the transformation to be applied to the images
    :return:
    """
    for key, value in path_dictionary.items():
        original_path = value['original_path']
        normalized_path = value['normalized_path']
        image = Image.open(original_path).convert('RGB')

        # apply resizing and normalization
        transformed_image = transform(image)

        # convert above tensor to image
        image_to_save = transforms.ToPILImage()(transformed_image)
        # save image in the normalized directory
        image_to_save.save(normalized_path)


cleanup_root()

# path to the zip file
zip_file_path = 'data/dataset.zip'

extract_zip(zip_file_path)

# dictionary to store the paths of the original and normalized images
dataset_dict = {}

# rename files, this will also update the dictionary
labels = ['angry', 'happy', 'neutral', "focused"]

for label in labels:
    rename_files(f'data/{label}', f'{label}')
# save the dictionary to a json file to be used later by other modules
with open('dataset.json', 'w') as json_file:
    json.dump(dataset_dict, json_file, indent=4)

# resize, make tensor, normalize
transformation = transforms.Compose([
    transforms.Resize((224, 224)),  # for resnet18 for example
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # standard normalization values
])

make_normalized_directories(labels)

generate_transformed_images(dataset_dict, transformation)
