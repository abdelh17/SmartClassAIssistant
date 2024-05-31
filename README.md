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
├── datavisualization.py
├── dataset.json
└── requirements.txt

```
The new folders contain images.  
The file ```dataset.json``` is a json file that contains path and label information about all images.

The file ```data_visualization.py``` contains the code that handles the evisualization of the data. When running this file, a folder ```plots/``` will be created under the ```code/``` folder.


## Instructions to run the code
### Setup
1. Clone the repository.
2. In the root, execute the following commands in a terminal:
```bash
cd code
python -m venv venv
```
3. If running on Windows run the following:
```
venv\Scripts\activate
```
If running on MacOS or Linux run the following:
```
source venv/bin/activate
```
4. Install requirements
```
pip install -r requirements.txt
```
### Run the preprocessing
1. In the code directory, execute:
```
python preprocessing.py
```
### Run the data visualization
**Note: you must have ran the previous command at least once before. Your folder structure shoud look like above after executing preprocessing.py**
1. In the code directory, execute:
```
python data_visualization.py
```





