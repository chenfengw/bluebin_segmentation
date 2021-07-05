# Trash Bin Detection

This repository contains code for detecting blue trash bin using gaussian classifier.

## Examples
![](data/results/results1.png)|![](data/results/results2.png)
--|--
![](data/results/results3.png)|![](data/results/results4.png)

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s), run this command:

```train
python training.py
```

## Evaluation
To evaluate the model:

```eval
python test_bin_detector.py
```

## Modeling Distributions
We can use gaussian distribution to model the class distribution of a particular label. y = {blue, not blue}
![](images/eq1.png)

Using bayes rule, we can write the joint probability as
![](images/eq2.png)

Therefore the posterior can be written as
![](images/eq3.png)

Our classifier is therefore in the form:
![](images/eq4.png)
