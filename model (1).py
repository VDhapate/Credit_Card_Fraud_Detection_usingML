import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
import pickle

# Load dataset
print("📂 Loading dataset...")
dataset = pd.read_csv(r"C:\Users\hp\OneDrive\Documents\BlackBox\dataset3.csv")
print(dataset.head())

# Drop unused columns
dataset.drop(['transaction_id', 'customer_id', 'location', 'customer_age', 'fraud_type'], axis=1, inplace=True)

# Convert time and extract features
dataset['transaction_time'] = pd.to_datetime(dataset['transaction_time'])
dataset['hour'] = dataset['transaction_time'].dt.hour
dataset['day_of_week'] = dataset['transaction_time'].dt.dayofweek
dataset['month'] = dataset['transaction_time'].dt.month
dataset.drop(['transaction_time'], axis=1, inplace=True)

# Handle missing values
dataset['merchant_id'] = dataset['merchant_id'].fillna(dataset['merchant_id'].mode()[0])
dataset['amount'] = dataset['amount'].fillna(dataset['amount'].median())
for col in ['card_type', 'purchase_category']:
    dataset[col] = dataset[col].fillna('Unknown')
dataset.dropna(subset=['is_fraudulent'], inplace=True)

# One-hot encoding
dataset = pd.get_dummies(dataset, columns=['card_type', 'purchase_category'])

# Split features and target
X = dataset.drop('is_fraudulent', axis=1)
y = dataset['is_fraudulent']

# Fill any missing values in X
if X.isnull().sum().sum() > 0:
    print("🚨 Found missing values in X. Filling with 0s.")
    X.fillna(0, inplace=True)

# Balance dataset using SMOTE
print("🔄 Applying SMOTE to balance dataset...")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Train models
models = {
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced')),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, class_weight='balanced')),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

# Select and save best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name} with Accuracy: {accuracies[best_model_name]:.4f}")

# Save model and feature list
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# --- User Input Prediction ---
def get_user_input(feature_columns):
    print("\n📝 Enter transaction details:")
    try:
        input_data = {
            'amount': float(input("Amount: ")),
            'merchant_id': int(input("Merchant ID: ")),
            'hour': int(input("Hour (0-23): ")),
            'day_of_week': int(input("Day of Week (0=Mon, 6=Sun): ")),
            'month': int(input("Month (1-12): "))
        }
    except ValueError:
        print("🚫 Invalid numeric input.")
        return None

    # One-hot encoding inputs
    selected_card_type = input("Card Type (Credit, Debit, Rupay, MasterCard, Unknown): ").strip().lower()
    selected_purchase = input("Purchase Category (Food, Shopping, Digital, Unknown): ").strip().lower()

    # Convert to one-hot column names
    card_type_key = f'card_type_{selected_card_type}'
    purchase_key = f'purchase_category_{selected_purchase}'

    # Set initial zeros for all one-hot features
    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0

    # Set selected category columns to 1 if they exist
    for field in [card_type_key, purchase_key]:
        if field in feature_columns:
            input_data[field] = 1

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return input_df

# Prediction function
def predict_with_threshold(model, input_df, threshold=0.35):
    proba = model.predict_proba(input_df)[0][1]
    print(f"🔍 Fraud Probability: {proba:.4f} (Threshold: {threshold})")
    return 1 if proba >= threshold else 0

# Prediction loop
feature_columns = X.columns
while True:
    user_input_df = get_user_input(feature_columns)

    if user_input_df is not None:
        prediction = predict_with_threshold(best_model, user_input_df, threshold=0.35)
        if prediction == 1:
            print("\n⚠️ Prediction: Fraudulent Transaction")
        else:
            print("\n✅ Prediction: Legitimate Transaction")
    else:
        print("🚫 Prediction aborted due to invalid input.")

    again = input("\n🔁 Do you want to make another prediction? (yes/no): ").strip().lower()
    if again != 'yes':
        print("👋 Exiting prediction loop. Goodbye!")
        break
