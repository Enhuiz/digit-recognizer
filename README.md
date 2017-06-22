# Digit Recognizer - Convolutional Neural Network in Tensorflow

## How to run

### 1. Create a directory called data and copy your data downloaded from Kaggle into it

```
$ mkdir data
$ cp path-to-your-download-directory/train.csv data/train.csv
$ cp path-to-your-download-directory/test.csv data/test.csv
```

### 2. Install dependencies

```
$ pip3 install numpy
$ pip3 install pandas
$ pip3 install tensorflow
```

### 3. Train your model

```
python3 main.py
```

## FYI

It will take a very long time if you trying to train CNN using CPU. To enable GPU for faster training, you need a NVIDIA graphic card that supports cuda. The configurations are kind of complicated and you can find some instructions [here](https://www.tensorflow.org/install/).
