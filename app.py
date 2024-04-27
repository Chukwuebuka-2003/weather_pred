import streamlit as st
from joblib import load
import pandas as pd

# Load the saved XGBRegressor models
loaded_models = {}
for target_column in ['Minimum Temperature', 'Wind Speed', 'Wind Direction', 'Maximum Temperature', 'Relative Humidity', 'Rainfall']:
    loaded_models[target_column] = load(f'{target_column}_xgb_model.sav')

# Streamlit app
st.title("Rainfall Prediction")

st.write("Enter the year and month to predict the weather conditions:")

# Input fields for year and month
year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
month = st.selectbox("Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

# Convert month to numerical representation
months_map = {
    'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5, 
    'July': 6, 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11
}
month_numeric = months_map[month]

# Sample data for prediction based on user input
user_input_data = pd.DataFrame({
    'Year': [year],
    'Month': [month_numeric]
})

# Predict using the loaded models
predicted_values = {}
for target_column, model in loaded_models.items():
    predicted_values[target_column] = model.predict(user_input_data)[0]

# Display the predicted values
st.write("Predicted weather conditions:")
for target_column, value in predicted_values.items():
    st.write(f"{target_column}: {value}")
