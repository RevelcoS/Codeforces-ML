# Codeforeces problem rating prediction

This is a small servive, that can predict the problem rating based on the statement.

## Model

This model uses BERT for text embedding, and NN composed with Linear layers for the rating regression.

The model is automatically saved in `saves` directiory, and also can be trained with incremental learning.

For the train and test details see the `Run` section.

## Prediction

When you trained the model, you can either test it with samples from `samples` directory, or use the web application to make predictions in real time.

For the details, also see the `Run` section.

## Setup

To run the project, you should first setup the python environment, install the dependencies and download the dataset from the hugging face website.

### Environment

It is highly recommended to have the python environment to not mix up globally downloaded libraries.

First, `cd` into the project root directiory, and run
```
python3 -m venv env
source env/bin/activate
```
This above commands are for UNIX-based systems. For Windows setup the commands are similar. You can read the docs from official python [website](https://docs.python.org/3/library/venv.html) for details.

### Dependencies

To install the dependencies, run
```
pip3 install -r requirements.txt
```

### Dataset

To download the dataset, run
```
python3 download.py
```

## Run

First you should train the model, and then you can test it on samples or in the web service.

### Train
To train the model, run
```
python3 src/model.py --train
```
You can also specify the number of epochs, batch size, and train/test data size with the corresponding flags.

### Test
To test the model on samples, run
```
python3 src/model.py --test
```

### Web
To launch the web service locally, just run
```
python3 src/app.py
```

You can also run the service with the Dockerfile
```
docker build -t codeforces .
docker run -it codeforces
```
