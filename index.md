# Quora-Question-Pairs

This repository contains the code for our submission in Kaggle's competition Quora Question Pairs 
in which we ranked in the top 25%. A detailed report for the project can be found [here](https://drive.google.com/file/d/0B76BuJcKjuxqM0tVOXd1cVVXb1k/view?usp=sharing)

## Data

`train.csv` contains ~ 400k question pairs along with the corresponding label (duplicate or not) and 
`test.csv` contains ~ 2300k question pairs. Both the files can be found [here](https://www.kaggle.com/c/quora-question-pairs/data).

## Model Architecture

We use a [Siamese Neural Network](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf) architecture with [Gated Recurrent Units](https://arxiv.org/abs/1409.0473) in combination with 
traditional Machine Learning algorithms like Random Forest, SVM and Adaboost. (Arxiv paper soon to be 
linked with results).

## Running the model

Firstly, place the `train.csv`,`test.csv` (see the **Data** section above) and the pre-trained GloVe embeddings in the `input` folder. You can download the embeddings from [here](https://drive.google.com/open?id=0B76BuJcKjuxqZG5YdG5SekU0VFk). Then, simply run the bash script:

```bash
bash run_model.sh
```

## Contributors
- [Ameya Godbole](https://github.com/sabersword)
- [Aman Dalmia](https://github.com/dalmia)

## Dependencies
- numpy
- pandas
- nltk
- sklearn
- TensorFlow

Install them using [pip](https://pypi.python.org/pypi/pip). 
