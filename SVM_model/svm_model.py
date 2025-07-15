import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import LabelEncoder

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

# Train
mod = svm.SVC(kernel="rbf", C=1.0, gamma="scale")
mod.fit(x_train, y_train)

# Evaluation
y_pred = mod.predict(x_test)
print(classification_report(y_test, y_pred))
joblib.dump(mod, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")