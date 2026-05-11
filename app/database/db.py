import sqlite3

DB_PATH = "emails.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn