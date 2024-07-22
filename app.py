from typing import Dict
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from os.path import dirname, join, realpath
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from encoders import *
# Load the trained model using st.cache_resource
@st.cache_resource
def load_model(model_name):
    with open(join(dirname(realpath(__file__)), model_name), "rb") as f:
        model = joblib.load(f)
    return model

model = load_model("MODEL/greens.pkl")

# Function to preprocess input data
def preprocess_data(mothersage, typeplaceofresidence, education, wealth, ancvisits, deliverybyCS, 
                    twbirthinlast3years, twinchild, drugsforintestinalparasites):
    # Create DataFrame from user input
    data = {
        'mothersage': [mothersage],
        'typeplaceofresidence': [typeplaceofresidence],
        'education': [education],
        'wealth': [wealth],
        'ancvisits': [ancvisits],
        'deliverybyCS': [deliverybyCS],
        'twbirthinlast3years': [twbirthinlast3years],
        'twinchild': [twinchild],
        'drugsforintestinalparasites': [drugsforintestinalparasites]
    }
    # Perform any necessary data preprocessing (e.g., encoding categorical variables)
    # Example: One-hot encode categorical variables
    def encode_label(label, dictionary: Dict):

        return dictionary.get(label)
    return [encode_label(mothersage, age_mapping),encode_label(typeplaceofresidence, typeplaceofresidence_mapping), 
            encode_label(education, education_mapping),
            encode_label(wealth, wealth_mapping),
            float(ancvisits),encode_label(deliverybyCS, deliverybyCS_mapping),
            float(twbirthinlast3years),
            encode_label(twinchild, twinchild_mapping),
            encode_label(drugsforintestinalparasites, drugsforintestinalparasites_mapping),
             ]

# Streamlit app code
st.title("Low Birth Weight Prediction Application")
#st.image("LBW.jpg")
st.subheader("A simple app to predict whether a baby is likely to have low birth weight.")

# Form to collect user information
my_form = st.form(key="LBW_form")
mothersage = my_form.selectbox("Select the age range of the mother", ["15-19", "20-24", "25-29", "30-34", "35-49"])
typeplaceofresidence = my_form.selectbox("Select the type of place of residence", ["Urban", "Rural"])
education = my_form.selectbox("Select the highest level of education", ["No education", "Primary", "Secondary", "Higher"])
wealth = my_form.selectbox("Select the wealth index", ["Poorest", "Poorer", "Middle", "Richer", "Richest"])
ancvisits = my_form.number_input("Enter the number of ANC visits", min_value=0, max_value=20)
deliverybyCS = my_form.selectbox("Select if the delivery is by caesarean section", ["No", "Yes"])
twbirthinlast3years = my_form.number_input("Enter the number of births in the last three years", min_value=0, max_value=10)
twinchild = my_form.selectbox("Select if the child is a twin", ["Single birth", "2nd of multiple"])
drugsforintestinalparasites = my_form.selectbox("Select if drugs for intestinal parasites were taken during pregnancy", ["No", "Yes"])

submit = my_form.form_submit_button(label="Predict")

if submit:
    # Preprocess the input data
    input_data = preprocess_data(mothersage, typeplaceofresidence, education, wealth, ancvisits, deliverybyCS, 
                                 twbirthinlast3years, twinchild, drugsforintestinalparasites)

    # Perform prediction using the loaded model
    prediction = model.predict((input_data,))
    prediction_proba = model.predict_proba((input_data,))

    # Display prediction result
    st.header("Prediction Result")
    if prediction[0] == 1:
        st.write("The model predicts that the baby is likely to have low birth weight.")
    else:
        st.write("The model predicts that the baby is not likely to have low birth weight.")

    st.subheader("Prediction Probabilities")
    st.write(f"Probability of low birth weight: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of normal birth weight: {prediction_proba[0][0]:.2f}")

# Display footer
st.write("Developed with ❤️😍 by Group5")
