# Quora-Question-Pairs

This repository contains the code for our submission in Kaggle's competition Quora Question Pairs 
in which we ranked in the top 25%.

## Data

`train.csv` contains ~ 400k question pairs along with the corresponding label (duplicate or not) and 
`test.csv` contains ~ 2300k question pairs. Both the files can be found [here](1).

## Model Architecture

We use a [Siamese Neural Network](2) architecture with [Gated Recurrent Units](3) in combination with 
traditional Machine Learning algorithms like Random Forest, SVM and Adaboost. (Arxiv paper soon to be 
linked with results).

![Architecture](https://github.com/dalmia/Quora-Question-Pairs/blob/master/architecture.png)

## Running the model

Firstly, place the `train.csv`,`test.csv` (see the **Data** section above) and the pre-trained GloVe embeddings in the `input` folder. You can download the embeddings from [here](5). Then, simply run the bash script:

```bash
bash run_model.sh
```

## Dependencies
- numpy
- pandas
- nltk
- sklearn
- TensorFlow

Install them using [pip](4).

[1]: https://www.kaggle.com/c/quora-question-pairs/data
[2]: http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf
[3]: https://arxiv.org/abs/1409.0473
[4]: https://pypi.python.org/pypi/pip
[5]: https://drive.google.com/open?id=0B76BuJcKjuxqZG5YdG5SekU0VFk
