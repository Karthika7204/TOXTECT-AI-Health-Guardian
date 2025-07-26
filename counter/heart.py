import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

def load_and_preprocess(file_path):
    """Load dataset, drop unnecessary columns, and encode gender."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    drop_cols = ["Diet_Score", "Cholesterol_Level", "Triglyceride_Level", 
                 "LDL_Level", "HDL_Level", "Systolic_BP", "Diastolic_BP",
                 "Air_Pollution_Exposure", "Family_History", "State_Name", 
                 "Emergency_Response_Time", "Annual_Income", "Health_Insurance", 
                 "Patient_ID"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df.dropna(inplace=True)
    X = df.drop("Heart_Attack_Risk", axis=1)
    y = df["Heart_Attack_Risk"]
    return X, y

def train_model(X, y):
    """Split the data, balance it using SMOTE, train XGBoost, and display metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return model

def predict_user_input(model):
    """Get user input for features, predict risk, and provide suggestions."""
    print("\nEnter the following details to predict heart attack risk:")
    try:
        age = int(input("Age: "))
        gender_input = input("Gender (Male/Female): ").strip().lower()
        if gender_input == 'male':
            gender = 1
        elif gender_input == 'female':
            gender = 0
        else:
            print("Invalid gender input. Please enter Male or Female.")
            return
        diabetes = int(input("Diabetes (1 for Yes, 0 for No): "))
        hypertension = int(input("Hypertension (1 for Yes, 0 for No): "))
        obesity = int(input("Obesity (1 for Yes, 0 for No): "))
        smoking = int(input("Smoking (1 for Yes, 0 for No): "))
        alcohol = int(input("Alcohol Consumption (1 for Yes, 0 for No): "))
        physical_activity = int(input("Physical Activity (1 for Yes, 0 for No): "))
        stress = int(input("Stress Level (0-4): "))
        healthcare_access = int(input("Healthcare Access (1 for Yes, 0 for No): "))
        heart_attack_history = int(input("Heart Attack History (1 for Yes, 0 for No): "))
    except ValueError:
        print("Invalid input. Please enter numeric values as instructed.")
        return
    features = [[age, gender, diabetes, hypertension, obesity, 
                 smoking, alcohol, physical_activity, stress, 
                 healthcare_access, heart_attack_history]]
    risk = model.predict(features)[0]
    if risk == 1:
        print("\nPrediction: High risk of heart attack.\n")
        print("Lifestyle Suggestions:")
        print("- Quit smoking and limit alcohol consumption.")
        print("- Engage in regular physical activity.")
        print("- Maintain a balanced diet rich in fruits and vegetables.")
        print("- Monitor and manage stress levels effectively.")
        print("- Attend regular health check-ups and follow medical advice.")
    else:
        print("\nPrediction: Low risk of heart attack.\n")
        print("Lifestyle Suggestions:")
        print("- Continue maintaining a healthy lifestyle.")
        print("- Engage in regular exercise and balanced diet.")
        print("- Avoid smoking and limit alcohol intake.")
        print("- Keep stress levels under control.")
        print("- Stay up to date with regular health screenings.")

if __name__ == "__main__":
    data_file = r"E:\Foodie-calorie-finder-main\counter\data\heart_attack_prediction_india.csv"
    X, y = load_and_preprocess(data_file)
    model = train_model(X, y)
    joblib.dump(model, 'heart_attack_model.pkl')
    model = joblib.load('heart_attack_model.pkl')
    predict_user_input(model)
