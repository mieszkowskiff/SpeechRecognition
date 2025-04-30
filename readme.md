# Speech Recognition Project

## Introduction

This project is a part of Summer Semester 2024/2025 Deeplearning course at Warsaw University of Technology, faculty of Mathematics and Information Sciences.

The aim of this project is to implement several deeplearning architectures (including transformers and CNNs) to classify basic speech commands.

## Dataset

The dataset can be found [here](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/46945?fbclid=IwZXh0bgNhZW0CMTEAAR5OByzx2ExdHvDP5ACWn8WsscWpd70PORfKu5J-D-SMwVmZCat6ja1ezkciBQ_aem_yMGkvJ5VDFUNLTSpOpnodw).

## Dependencies
```{Bash}
pip install torchaudio numpy matplotlib scikit-learn torchsummary
```

## Usage
After downloading the dataset run `split_dataset.py` to split dataset into train, valid and test part. (you can also run `create_unknown_class.py` earlier to merge classes into one). Select desired parameters in the `main.py` and run the file.
Models will be saved under `./models/`. You can later analyze the model using `create_confusion_matrix.py` or `transformed_spectrogram.py`.


