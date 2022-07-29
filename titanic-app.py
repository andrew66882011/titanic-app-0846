import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Titanic Survival Prediction App

This app predicts the **Survival** from the **Titanic Tragedy**!
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)[['Class', 'Sex', 'Age']]
else:
    def user_input_features():
        Class = st.sidebar.selectbox('Class',('1st','2nd','3rd', 'Crew'))
        Sex = st.sidebar.selectbox('Sex',('Male','Female'))
        Age = st.sidebar.selectbox('Age', ('Child', 'Adult'))
        data = {'Class': Class,
                'Sex': Sex,
                'Age': Age}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire Titanic dataset
# This will be useful for the encoding phase
titanic_raw = pd.read_csv('titanic.csv')
titanic = titanic_raw[['Class', 'Sex', 'Age']]
df = pd.concat([input_df,titanic],axis=0)

# Encoding of ordinal features
encode = ['Class', 'Sex','Age']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('titanic_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
survival = np.array(['No','Yes'])
st.write(survival[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
