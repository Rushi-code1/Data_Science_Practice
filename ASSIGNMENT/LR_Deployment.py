import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Titanic_train.csv')
    return df

df = load_data()

st.title("Titanic Survival Prediction")

# Sidebar for user inputs
st.sidebar.header('Passenger Features')
def user_input_features():
    pclass = st.sidebar.selectbox('Passenger Class', (1, 2, 3))
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    age = st.sidebar.slider('Age', 0, 80, 29)
    sibsp = st.sidebar.slider('Siblings/Spouses Aboard', 0, 8, 0)
    parch = st.sidebar.slider('Parents/Children Aboard', 0, 6, 0)
    fare = st.sidebar.slider('Fare', 0.0, 512.329, 32.204)
    embarked = st.sidebar.selectbox('Port of Embarkation', ('C', 'Q', 'S'))
    
    # Create a data dictionary
    data = {'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Preprocess the data and set up the model pipeline
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Encoding sex
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # Encoding embarked

# Handle missing data
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# Imputation and scaling pipeline
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Predict on user input
input_df['Sex'] = input_df['Sex'].map({'male': 0, 'female': 1})  # Apply same encoding
input_df['Embarked'] = input_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # Apply encoding
prediction = pipeline.predict(input_df)
prediction_proba = pipeline.predict_proba(input_df)

# Show the prediction result
st.subheader('Prediction')
survival = np.array(['Not Survived', 'Survived'])
st.write(survival[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
