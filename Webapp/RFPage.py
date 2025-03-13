import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




with open ('../Data/models/rfModel.pkl', 'rb') as f:
    models = pickle.load(f)

with open ('../Data/cleanedDF.pkl', 'rb') as f:
    data = pickle.load(f)


df = data.copy()
rf_classifier = models['RandomForest']
accuracy = models['RandomForestAccuracy']
report = models['RandomForestReport']

st.title('Random Forest Model to predict Heart Attack')

st.write('This page will show the results of our Random Forest model')

st.write('The model was trained on the following data:')
st.write(df.head())

st.write('The model was trained on the following features:')
st.write(df.columns)

st.write('Classification Report:')
st.write(report)

st.write('Accuracy:')
st.write(accuracy)


