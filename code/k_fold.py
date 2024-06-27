# k_fold.py
"""
This script performs K-fold cross-validation on a convolutional neural network (CNN) architecture
for image classification. It combines the functionality of setting up the dataset and performing
the K-fold cross-validation training and evaluation.

Key functionalities:
1. Fetch data from the dataset directory and assign labels to each class.
2. Save the full dataset and K-fold indices to pickle files for future use.
3. Load the dataset and K-fold indices.
4. Perform K-fold cross-validation (10 folds) to train and evaluate the models.
5. Use early stopping based on validation loss during training.
6. Calculate and print performance metrics (accuracy, precision, recall, F1-score) for each fold and average across all folds.

Requirements:
- Dataset directory should contain subdirectories for each class ('angry', 'focused', 'happy', 'neutral') with images.
- Models (SimpleCNN, Model2, Model3) and helper functions (DatasetLoader, get_device, calculate_accuracy) are defined in the 'cnn_model' and 'Dataset' modules.
"""

import os
import pickle
import zipfile
from sklearn.model_selection import KFold, train_test_split
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim, nn
from sklearn.metrics import precision_score, recall_score, f1_score
from Dataset import DatasetLoader, get_device, calculate_accuracy  # Adjusted import names
from cnn_model import SimpleCNN, Model2, Model3


def fetchData(data_dir):
    """
    Fetches data from the specified directory and assigns labels to each class.

    Returns:
        all_images (list): List of image file paths.
        cls_label (list): List of corresponding class labels.
    """
    all_images = []
    cls_label = []

    # Get all images and labels
    for idx, cls in enumerate(['angry', 'focused', 'happy', 'neutral']):
        Cpath = os.path.join(data_dir, cls)
        if not os.path.exists(Cpath):
            raise FileNotFoundError(f"The directory {Cpath} does not exist. Please check the dataset path.")
        files = os.listdir(Cpath)
        for img in files:
            all_images.append(os.path.join(Cpath, img))
            cls_label.append(idx)

    return all_images, cls_label


if __name__ == '__main__':
    # Paths
    dataset_zip_path = 'data/dataset.zip'  # Path to the dataset zip file
    dataset_dir = 'data/dataset'  # Path to extract the dataset

    # Ensure dataset directory exists
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Extract the dataset if it doesn't already exist
    if not os.listdir(dataset_dir):
        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

    # Fetch images and labels
    images, labels = fetchData(dataset_dir)

    splits = {
        'images': images,
        'labels': labels,
    }

    # Save all images and labels to a pickle file
    with open('full_dataset.pkl', 'wb') as f:
        pickle.dump(splits, f)

    # Setting up K-fold cross-validation with 10 folds
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    # Split the dataset into folds
    splits2 = list(kfold.split(images))

    # Save all folds to a pickle file for future use
    with open('kfold_dataset.pkl', 'wb') as f:
        pickle.dump(splits2, f)

    # Hyperparameters and settings
    batch_size = 35
    test_batch_size = 35
    epochs = 20
    patience = 3

    # Load the full dataset from the pickle file
    with open('full_dataset.pkl', 'rb') as f:
        splits = pickle.load(f)
    X = splits['images']
    Y = splits['labels']

    # Load the K-fold indices from the pickle file
    with open('kfold_dataset.pkl', 'rb') as f:
        folds = pickle.load(f)

    # Get the device (CPU or GPU) for training
    device = get_device()
    print("Using device: ", device)

    # List of models to be trained
    models = [SimpleCNN(), Model2(), Model3()]

    # List to store metrics for each fold
    fold_metrics = []

    for i, (train_index, test_index) in enumerate(folds):
        print(f'------------------------------\nFold {i + 1}\n------------------------------')

        # Get the data for the current fold
        train_index = np.array(train_index, dtype=int)
        test_index = np.array(test_index, dtype=int)
        x_temp = [X[m] for m in train_index]
        x_test = [X[m] for m in test_index]
        y_temp = [Y[m] for m in train_index]
        y_test = [Y[m] for m in test_index]

        # Split the training data into training and validation sets
        x_train, x_valid, y_train, y_valid = train_test_split(x_temp, y_temp, test_size=0.15, random_state=42,
                                                              stratify=y_temp)

        # Create DataLoaders for training, validation, and testing
        trainset = DatasetLoader(x_train, y_train)  # Adjusted class name as needed
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

        testset = DatasetLoader(x_test, y_test)  # Adjusted class name as needed
        test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=8, drop_last=True)

        validset = DatasetLoader(x_valid, y_valid)  # Adjusted class name as needed
        valid_loader = DataLoader(validset, batch_size=test_batch_size, shuffle=False, num_workers=8, drop_last=True)

        for model in models:
            model_name = model.__class__.__name__
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            best_val_loss = float('inf')
            patience_counter = 0

            # Training loop
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                train_accuracy = calculate_accuracy(train_loader, model)

                model.eval()
                val_loss = 0.0
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                val_accuracy = calculate_accuracy(valid_loader, model)

                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, '
                      f'Val Loss: {val_loss / len(valid_loader):.4f}, '
                      f'Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')

                # Early stopping based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'saved_models/best_model_{model_name}_fold_{i}.pth')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping, stopped at epoch: ", epoch + 1)
                    break

            # Evaluate on the test set using the model with the best validation loss
            model.load_state_dict(torch.load(f'saved_models/best_model_{model_name}_fold_{i}.pth'))
            model.eval()
            test_correct = 0
            test_total = 0
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            test_accuracy = 100 * test_correct / test_total
            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')

            fold_metrics.append({
                'fold': i + 1,
                'accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

            print(f'Test Accuracy: {test_accuracy:.2f}%\n'
                  f'Precision: {precision:.2f}\n'
                  f'Recall: {recall:.2f}\n'
                  f'F1-Score: {f1:.2f}\n\n')

    # Calculate average metrics across all folds
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    avg_precision = np.mean([m['precision'] for m in fold_metrics])
    avg_recall = np.mean([m['recall'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1_score'] for m in fold_metrics])

    print('------------------------------')
    print('Overall Performance')
    print('------------------------------')
    print(f'Average Accuracy: {avg_accuracy:.2f}%')
    print(f'Average Precision: {avg_precision:.2f}')
    print(f'Average Recall: {avg_recall:.2f}')
    print(f'Average F1-Score: {avg_f1:.2f}')
