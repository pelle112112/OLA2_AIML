# Ola 2

Pelle Hald Vedsmand and Nicolai Rosendahl

For this project we have selected a Heart Attack dataset, that should have features affecting the possibility of Heart attack.

The documentation of our selection of models, overfitting, hyperparameters and measurements of quality is at the bottom of the modeling.ipynb file.

We want to use the features for training of 3 models:

- Decision Tree
- Random Forest
- Naive Bayes

The cleaning of the data can be found in /Code/dataCleaning.ipynb, which saves the cleaned dataframe into a pickle file for later use.
The modeling and documentation of the models can be found in /Code/modeling.ipynb

The presentation of the measuring of the distances between datapoints is in the root of the project "presentation.pdf"

### To run the streamlit application

We have saved the models in pickle files, to be used in some kind of demo application. We have therefore created a streamlit application, which contains some info about the models, to showcase how we can export the models and use them elsewhere.

To run the streamlit application you need to install the modules from requirements.txt

```
pip install -r requirements.txt
```

You then need to generate the models.
Therefore you need to run the following Jupiter notebooks:

```
dataCleaning.ipynb
modeling.ipynb
```

Now you should have the models generated in Data/models

Then navigate to webapp

```
cd /Webapp
```

Finally run the streamlit application from the webapp folder

```
py -m streamlit run start.py
```
