# check_db.py
import sqlite3
import os

print(f"Current directory: {os.getcwd()}")
print(f"Database file exists: {os.path.exists('amac_users.db')}")

if os.path.exists('amac_users.db'):
    conn = sqlite3.connect('amac_users.db')
    c = conn.cursor()
    
    # Check tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = c.fetchall()
    print(f"\nTables in database: {tables}")
    
    # Check users
    try:
        c.execute("SELECT * FROM users")
        users = c.fetchall()
        print(f"\nUsers in database ({len(users)}):")
        for user in users:
            print(user)
    except Exception as e:
        print(f"\nError reading users table: {e}")
    
    conn.close()
else:
    print("\nDatabase file does not exist!")
    