# vehicle_insurance_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
import joblib

# Load dataset
data = pd.read_csv('data.csv')
df = data.copy()

# Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male=1, Female=0
df['Vehicle_Damage'] = le.fit_transform(df['Vehicle_Damage'])  # Yes=1, No=0

# Map vehicle age to ordered values
vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
df['Vehicle_Age'] = df['Vehicle_Age'].map(vehicle_age_map)

# Drop unnecessary columns
df = df.drop(['id', 'Gender', 'Driving_License'], axis=1)

# Split features and labels
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Handle imbalance
rs = RandomOverSampler()
x_resampled, y_resampled = rs.fit_resample(x, y)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=1)

# Standardize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Random Forest model
for d in [10, 15, 20]:
    for n in [100, 150]:
        rf = RandomForestClassifier(max_depth=d, n_estimators=n, random_state=42)
rf.fit(x_train, y_train)

# Predict
y_pred = rf.predict(x_test)
y_prob = rf.predict_proba(x_test)[:, 1]

# Evaluation
print("Random Forest Classifier")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("Train Accuracy:", rf.score(x_train, y_train))
print("Test Accuracy:", rf.score(x_test, y_test))

# Save model and scaler
joblib.dump(rf, 'random_forest_model.pkl', compress=3)
joblib.dump(scaler, 'scaler.pkl')







  




