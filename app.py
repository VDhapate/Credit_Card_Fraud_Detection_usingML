from flask import Flask, request, render_template, redirect
import sqlite3
import pickle
import pandas as pd
import os

app = Flask(__name__)

@app.route('/uploadfile', methods=['GET', 'POST'])
def uploadfile():
    if request.method == 'POST':
        try:
            amount = float(request.form['transactionAmount'])
        except ValueError:
            return "Invalid amount. Please enter a valid number.", 400
        category = request.form['transactionCategory']
        transaction_type = request.form['transactionType']
        merchant_id = request.form['Merchantname']
        date = request.form['transactionDate']
        day = request.form['transactionDay']
        month = request.form['transactionMonth']
        time = request.form['transactionTime']

        conn = sqlite3.connect('transactions.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL,
                category TEXT,
                transaction_type TEXT,
                merchant_id TEXT,
                date INTEGER,
                day INTEGER,
                month INTEGER,
                time INTEGER
            )
        ''')
        c.execute('''
            INSERT INTO transactions (amount, category, transaction_type, merchant_id, date, day, month, time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (amount, category, transaction_type, merchant_id, date, day, month, time))
        conn.commit()
        conn.close()
        return redirect('/predict_last')
    return render_template('uploadfile.html')

@app.route('/predict_last')
def predict_last():
    # Check if model files exist
    if not (os.path.exists('model.pkl') and os.path.exists('feature_columns.pkl')):
        return "Model files not found. Please train your model first."

    # Load model and feature columns
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)

    # Get the last transaction from the database
    conn = sqlite3.connect('transactions.db')
    c = conn.cursor()
    c.execute('SELECT amount, category, transaction_type, merchant_id, date, day, month, time FROM transactions ORDER BY id DESC LIMIT 1')
    row = c.fetchone()
    conn.close()

    if not row:
        return "No transaction found."

    # Prepare input dict
    input_dict = {
        'amount': float(row[0]),
        'merchant_id': int(row[3]),
        'hour': int(row[7]),  # 'time' is hour
        'day_of_week': int(row[5]),  # 'day'
        'month': int(row[6])
    }

    # One-hot encoding for category and transaction_type
    for col in feature_columns:
        if col not in input_dict:
            input_dict[col] = 0

    cat_key = f"purchase_category_{row[1].lower()}"
    type_key = f"card_type_{row[2].lower()}"
    if cat_key in feature_columns:
        input_dict[cat_key] = 1
    if type_key in feature_columns:
        input_dict[type_key] = 1

    # Prepare DataFrame
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if proba >= 0.35 else 0

    result = f"Fraud Probability: {proba:.2f} | Prediction: {'Fraudulent' if prediction else 'Legitimate'}"
    return render_template('result.html', result=result)

@app.route('/transactions')
def transactions():
    conn = sqlite3.connect('transactions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM transactions')
    rows = c.fetchall()
    conn.close()
    return render_template('transactions.html', transactions=rows)

@app.route('/success')
def success():
    return "Transaction uploaded successfully!"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('registration.html')

if __name__ == '__main__':
    app.run(debug=True)