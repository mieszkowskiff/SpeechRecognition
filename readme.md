# Speech Recognition Project

## Introduction

This project is a part of Summer Semester 2024/2025 Deeplearning course at Warsaw University of Technology, faculty of Mathematics and Information Sciences.

The aim of this project is to implement several deeplearning architectures (including transformers and CNNs) to classify basic speech commands.

## Prepare the dataset

Since this dataset comes from a Kaggle competition, the official test data is provided without labels. Therefore, this project only uses the `train` data and splits it into `train`, `validation`, and `test` sets. You are, of course, more than welcome to use the unlabeled `test` data provided by Kaggle for your own experiments.

1. **Download the dataset**  
The dataset can be found [here](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data). You need to find the download button (usually in the section `Data`, at the bottom)
2. **Extract the dataset**  
Extract the downloaded archive. Inside, you will find a `train.7z` archive. Extract it as well.
3. **Copy the `audio/` folder**  
Inside the extracted `train/` folder, you will find an `audio/` directory. Move this `audio/` directory to the root folder of this repository. You can now safely remove the rest of the downloaded archive.
4. **Create `unknown` class (optional)**
Run `create_unknown_class.py`
5. **Run `split_dataset.py`**  
Run the script to split the dataset. The processed files will be saved under a new `dataset/` directory. Once the process is complete, you can remove the original `audio/` directory to save disk space.


## Setup

**Linux**:
```{Bash}
python3 -m venv venv
source venv/bin/activate
pip install torchaudio numpy matplotlib scikit-learn torchsummary torchcodec
```

Ensure you have `ffmpeg` installed. If not, run:
```{bash}
sudo apt update
sudo apt install ffmpeg
```

## Usage

After downloading the dataset run `split_dataset.py` to split dataset into train, valid and test part. (See section Prepare the dataset)

### Training

Select desired parameters in the `train.py` and run the file.
Models will be saved under `./models/`, each model in separate directory.
Model (and optimizer) will be saved afer each epoch.

### Create confusion matrix

To create confusion matrix run `create_confusion_matrix.py`.
First select model (path and iteration) inside the `create_confusion_matrix.py` file.
Confusion matrix will be saved under `images` directory.

**Example confusion matrix:**
<img title="Confusion matrix" alt="Confusion matrix" src="./images/confusion_matrix.png">

### Create learning curve

To create learning curve run `create_learning_curve.py`.
First select model inside the `create_learning_curve.py` file.
Learning curve will be saved under `images` directory.

**Example learning curve**
<img title="Learning curve" alt="Learning curve" src="./images/learning_curve.png">

### Create spectorgram

You can also create a spectrogram (picture) of a given sound file. This is of course independent of deeplearning. To do so, run `create_spectrogram.py` (you can first select picture id inside it). It will also be saved inside `images/` directory.

**Example spectrogram**
<img title="Learning curve" alt="Learning curve" src="./images/spectrogram.png">

## Results
Experiments were conducted trying to determine optimal number of bloks and number of attention heads in each block. The results are presented below (big number is the f1 score, small number below is the number of parameters).
<img title="Learning curve" alt="Learning curve" src="./images/f1s_matrix.png">
