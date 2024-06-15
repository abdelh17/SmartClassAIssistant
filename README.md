# SmartClassAIssistant
## Group Members:

| Name | Student ID         |
|------------|--------------|
| Abdelkader Habel      |  40209153  |
| Abdelmalek Anes       | 40229242   |
|     Mohammed Afif Rahman   | 40203098|

## Contents
This repository contains a ```code``` folder. The hierarchy in it is as follows:
```
code/
├── data/
│   └── dataset.zip
├── preprocessing.py
├── cnn_model.py
├── Dataset.py
├── evaluation.py
├── inference.py
├── train_model.py
├── data_visualization.py
└── requirements.txt

```
In ```data/```, there is originally only the ```dataset.zip``` file. When the program runs, this folder will populate with the extracted folders in the archive and with a folder for normalized pictures.

The file ```preprocessing.py``` contains the code that handles the extraction, renaming, and processing of the data. When running this file, the structure of the ```code``` folder will look like follows:

```
code/
├── data/
│   ├── dataset.zip
│   ├── angry/
│   ├── happy/
│   ├── neutral/
│   ├── focused/
│   └── normalized_data/
├── preprocessing.py
├── cnn_model.py
├── Dataset.py
├── evaluation.py
├── inference.py
├── train_model.py
├── datavisualization.py
├── dataset.json
└── requirements.txt

```
The new folders contain images.  
The file ```dataset.json``` is a json file that contains path and label information about all images.

The file ```data_visualization.py``` contains the code that handles the evisualization of the data. When running this file, a folder ```plots/``` will be created under the ```code/``` folder.


## Instructions to run the code
### Setup
In a terminal:
1. Clone the repository.
```
git clone https://github.com/abdelh17/SmartClassAIssistant.git
cd SmartClassAIssistant/
```
2. In the root, execute the following commands in a terminal to create a virtual environment:
```bash
cd code
python -m venv venv
```
3. If running on Windows run the following:
```bash
venv\Scripts\activate
```
If running on MacOS or Linux run the following:
```bash
source venv/bin/activate
```
4. Install requirements
```bash
pip install -r requirements.txt
```
### Run the preprocessing
1. In the code directory, execute:
```bash
python preprocessing.py
```
### Run the data visualization
**Note: you must have ran the previous command at least once before. Your folder structure shoud look like above after executing preprocessing.py**
1. In the code directory, execute:
```bash
python data_visualization.py
```

### Run the training code
**Note: you must have ran the preprocessing command at least once before.**
1. In the code directory, execute:
```bash
python train_model.py
```
After the training code is run, a new folder is created at the root: ```saved_models```. It contains the three models and their weights.

### Run the evaluation code
**Note: you must have ran the train_model command at least once before. Your folder structure should have a ```saved_models``` folder.**
1. In the code directory, execute:
```bash
python evaluation.py
```
This will generate 3 confusion matrices in ```confusion_matrix/```, one for each model.


### Run the inference code
**Note: you must have ran the train_model command at least once before. Your folder structure should have a ```saved_models``` folder.**
1. In the code directory, execute:
```bash
python inference.py path_to_your_image_1 path_to_your_image_2 path_to_your_image_3
```
**Replace the above paths by the images you want to make an inference on. At least one path is needed, and there is no limit to the number of images passed.**  
Example:
```bash
python inference.py data/angry/angry234.jpg data/happy/happy45.jpg data/neutral/neutral34.jpg
```

## When done
To exit the virtual environment execute:
```bash
deactivate
```

## Link to the full dataset
https://www.kaggle.com/datasets/msambare/fer2013



