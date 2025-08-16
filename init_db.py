import sqlite3

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
conn.commit()
conn.close()