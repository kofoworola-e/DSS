"""
This is a simple Streamlit prediction app'
"""

import streamlit as st
import joblib
import pandas as pd
from preprocessing import preprocess_user_input

# Load the trained model
model = joblib.load('random_forest_model.pkl')


# Define a function to handle user input
def get_user_input():
    """
    Collects user input from the Streamlit interface for
     credit card approval prediction.

    The function creates a user-friendly interface to gather 
    information about the applicant,
    including personal details and socioeconomic factors.
    The collected inputs are organized into a pandas DataFrame
    for further processing.

    Returns:
        pd.DataFrame: A DataFrame containing the user's inputs, structured
                      for model prediction.
    """    
    gender = st.selectbox('Gender', ['Female', 'Male'])
    car_owner = st.selectbox('Car Owner', ['Yes', 'No'])
    property_owner = st.selectbox('Property Owner', ['Yes', 'No'])
    housing_type = st.selectbox('Housing Type', ['House', 'Apartment',
                                                 'Office Apartment'])
    occupation = st.selectbox('Occupation', ['Administrative Staff',
                                             'Service Staff',
                                             'Laborers',
                                             'Tech Staff',
                                             'Specialized Staff',
                                             'Medical Staff'])
    # Ordinal features
    age_group = st.selectbox('Age Group', ['20-29', '30-38', '39-46', '47-64'])

    income_category = st.selectbox('Income Category', ['Low Income',
                                                       'Middle Income',
                                                       'High Income',
                                                       'Very High Income'])

    family_size = st.selectbox('Family Size', ['1-2', '3-4', '5+'])
    income_type = st.selectbox('Income Type', ['Employee', 
                                               'State Employee', 
                                               'Pensioner'])
    education = st.selectbox('Education', ['Secondary Education', 
                                           'Higher Education'])
    communication_access = st.selectbox('Communication Access', [
                                                    'Limited Access',
                                                    'Full Access'])

    # Create a DataFrame from the input data
    user_data = pd.DataFrame({
        'gender': [gender],
        'age_group': [age_group],
        'car_owner': [car_owner],
        'property_owner': [property_owner],
        'housing_type': [housing_type],
        'family_size': [family_size],
        'communication_access': [communication_access],
        'education': [education],
        'income_type': [income_type],
        'occupation': [occupation],
        'income_category': [income_category]
    })
    # Convert all string inputs to lowercase
    user_data = user_data.apply(lambda x: x.str.lower() if x.dtype == 'object'
                                else x)

    return user_data


# Main app function
def main():
    """
    Main function to run the Streamlit app for credit card approval prediction.

    This function sets up the Streamlit application interface,
    displays the title and description, collects user input, 
    preprocesses the input data, and uses the trained Random Forest model 
    to make predictions on credit card approvals. The results of the 
    predictions are displayed to the user, indicating whether
    the credit card application is approved or not.

    Returns:
        None
    """
    st.title('Credit Card Approval Classifier')
    st.write("This app uses a Random Forest model to make predictions.")

    # Get user input
    user_input = get_user_input()

    # Add a 'Predict' button
    if st.button('Predict'):
        # Preprocess the user input
        user_input_processed = preprocess_user_input(user_input)

        # Predict using the model
        prediction = model.predict(user_input_processed)

        # Display the result
        if prediction[0] == 0:
            st.success("Credit Card Approved!")
        else:
            st.error("Credit Card Not Approved.")


if __name__ == '__main__':
    main()
