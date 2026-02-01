import sqlite3

db_path = "D:\archive\wikibooks.sqlite"

try: 
    connection = sqlite3.connect(db_path)
    print(f"Successfully connected to {db_path}")
except sqlite3.Error as e:
    print(f"Error - {e}")