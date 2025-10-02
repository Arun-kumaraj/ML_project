import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("Salary Data.csv").dropna()

# Features & target
X = data[['Years of Experience', 'Age', 'Gender', 'Education Level']]
y = data['Salary']

# Preprocessing for categorical data
categorical_features = ['Gender', 'Education Level']
numeric_features = ['Years of Experience', 'Age']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "salary_model.pkl")

# Test prediction
test_input = pd.DataFrame([[6, 30, 'Male', "Bachelor's"]], columns=['Years of Experience','Age','Gender','Education Level'])
print("Predicted salary:", model.predict(test_input)[0])
