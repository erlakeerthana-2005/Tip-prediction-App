import pandas as pd #to handle dataframes
from sklearn.model_selection import train_test_split #to split data
from sklearn.ensemble import RandomForestRegressor #to create model
from sklearn.preprocessing import OneHotEncoder #to handle categorical variables
from sklearn.compose import ColumnTransformer  #to apply transformations
from sklearn.pipeline import Pipeline  #to create pipeline
import joblib #to save model

def train_model(df):
    """train a machine learing model using the provided dataframes"""
    #separate tips and features as x and y
    X= df.drop(columns=['tip'])
    y=df['tip']

    #identify categorical and numerical colums
    categorial_cols = X.select_dtypes(include=['category']).columns
    numerical_cols = X.select_dtypes(exclude=['category']).columns

    #one hot encode categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('num','passthrough', numerical_cols),
            ('cat', OneHotEncoder(), categorial_cols)
        ])
    
    #create a pipeline with preprocessor and model
    model = Pipeline(
        steps=[
                ('preprocessor', preprocessor),
                    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
       ]
    )

    #split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #train the model
    model.fit(X_train, y_train) 
    return model