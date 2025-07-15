import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import svm
from sklearn.metrics import classification_report
import joblib
from sklearn.svm import SVC

# Load and clean data
df = pd.read_csv('data_assignment1.csv')
df.columns = df.columns.str.strip()
df = df.drop(columns=["Person ID"], errors="ignore")

# Categorical variables
label_encoder = LabelEncoder()
df["Sleep Disorder"] = label_encoder.fit_transform(df["Sleep Disorder"])

df = pd.get_dummies(df, columns=["Gender", "Age", "Occupation", "Sleep Duration", 
                                 "Quality of Sleep", "Physical Activity Level", "Stress Level", 
                                 "BMI Category", "Blood Pressure", "Heart Rate", "Daily Steps",])

# Features and Prediction(label)
x = df.drop(columns=["Sleep Disorder"])
y = df["Sleep Disorder"]

# Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=25)

# Grid Search
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1]
}

search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
search.fit(x_train, y_train)

print("Best Parameters:", search.best_params_)
print("Best Cross-validation Accuracy:", search.best_score_)