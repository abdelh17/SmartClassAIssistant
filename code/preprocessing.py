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


zip_file_path = 'data/dataset.zip'

extract_zip(zip_file_path)

# dictionary to store the paths of the original and normalized images
dataset_dict = {}

# rename files, this will also update the dictionary
rename_files('data/angry', 'angry')
rename_files('data/happy', 'happy')
rename_files('data/neutral', 'neutral')
# resize, make tensor, normalize
transformation = transforms.Compose([
    transforms.Resize((224, 224)),  # for resnet18 for example
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # standard normalization values
])

labels = ['angry', 'happy', 'neutral']

make_normalized_directories(labels)

generate_transformed_images(dataset_dict, transformation)