"""
processing.py

This module provides preprocessing functions for the
 Credit Card Approval Classifier app.
It includes functionality for encoding categorical 
 and ordinal features from user input data.

Functions:
- preprocess_user_input(user_input): 
    Preprocesses user input data by applying label encoding
     and ordinal encoding.
"""
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


def preprocess_user_input(user_input):
    """
    Preprocesses the user input data for the credit card approval model.

    This function takes the user input DataFrame, applies label encoding 
    to the categorical features, and transforms the ordinal features using 
    the pre-defined OrdinalEncoder. The processed input is then ready 
    for prediction by the model.

    Args:
        user_input (pd.DataFrame): A DataFrame containing user input 
                                   with categorical and ordinal features.

    Returns:
        np.ndarray: A NumPy array of the processed input data suitable
                    for model prediction.
    """
    # Define encoding details
    label_enc_features = ['gender', 'car_owner', 'property_owner', 
                          'housing_type', 'occupation']
    ordinal_features = ['age_group', 'income_category', 'family_size', 
                        'income_type', 'education', 'communication_access']

    # Initialize encoders
    ordinal_enc = OrdinalEncoder(categories=[
        ['20-29', '30-38', '39-46', '47-64'],
        ['low income', 'middle income', 'high income', 'very high income'],
        ['1-2', '3-4', '5+'],
        ['employee', 'state employee', 'pensioner'],
        ['secondary education', 'higher education'],
        ['limited access', 'full access']
    ])

    # Apply LabelEncoder to label encoding features
    for feature in label_enc_features:
        user_input[feature] = LabelEncoder().fit_transform(user_input[feature])

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal_enc', ordinal_enc, ordinal_features),
            ('label_enc', 'passthrough', label_enc_features) 
        ]
    )

    # Fit the preprocessor to user input
    preprocessor.fit(user_input)    

    # Transform the user input
    user_input_processed = preprocessor.transform(user_input)

    return user_input_processed
