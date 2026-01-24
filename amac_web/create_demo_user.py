# create_demo_user.py
import sqlite3
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_demo_user():
    conn = sqlite3.connect('amac_users.db')
    c = conn.cursor()
    
    # Create table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create demo user if not exists
    try:
        c.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("testuser", "demo@amac.com", hash_password("amac2024"))
        )
        conn.commit()
        print("✅ Demo user created: testuser / amac2024")
    except sqlite3.IntegrityError:
        print("⚠️  Demo user already exists")
    
    # Show all users
    c.execute("SELECT id, username, email FROM users")
    users = c.fetchall()
    print("\n📋 All users in database:")
    for user in users:
        print(f"  ID: {user[0]}, Username: {user[1]}, Email: {user[2]}")
    
    conn.close()

if __name__ == "__main__":
    create_demo_user()