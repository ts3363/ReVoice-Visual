from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import sqlite3
import hashlib
import os

app = FastAPI(title="AMAC Therapy")

# Templates
templates = Jinja2Templates(directory="templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
def init_db():
    conn = sqlite3.connect('amac_users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Include your existing API routers
# app.include_router(therapy_router, prefix="/api", tags=["therapy"])

app = FastAPI(title="AMAC Therapy")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ============ WEB PAGES ============
# Login page

@app.get("/login")
async def login_page():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AMAC Therapy - Login</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                height: 100vh; 
            }
            .login-box { 
                background: white; 
                padding: 40px; 
                border-radius: 15px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.2); 
                width: 380px; 
                text-align: center; 
            }
            h1 { 
                color: #2c5aa0; 
                margin-bottom: 8px; 
                font-size: 28px; 
            }
            .subtitle { 
                color: #666; 
                margin-bottom: 30px; 
                font-size: 16px; 
            }
            input { 
                width: 100%; 
                padding: 14px; 
                margin: 10px 0; 
                border: 2px solid #e1e5e9; 
                border-radius: 8px; 
                box-sizing: border-box; 
                font-size: 16px; 
                transition: border 0.3s;
            }
            input:focus { 
                outline: none; 
                border-color: #2c5aa0; 
                box-shadow: 0 0 0 3px rgba(44,90,160,0.1);
            }
            button { 
                width: 100%; 
                background: linear-gradient(135deg, #2c5aa0 0%, #1e3c72 100%); 
                color: white; 
                border: none; 
                padding: 16px; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 16px; 
                font-weight: 600; 
                margin-top: 15px; 
                transition: transform 0.2s;
            }
            button:hover { 
                transform: translateY(-2px); 
                box-shadow: 0 5px 15px rgba(44,90,160,0.3);
            }
            .demo { 
                background: #f7fafc; 
                border: 1px solid #e2e8f0; 
                border-radius: 8px; 
                padding: 15px; 
                margin-top: 25px; 
                font-size: 14px; 
                text-align: left; 
            }
            .links { 
                margin-top: 20px; 
                font-size: 14px; 
                color: #666;
            }
            a { 
                color: #2c5aa0; 
                text-decoration: none; 
                font-weight: 500;
            }
            a:hover { text-decoration: underline; }
            .error { 
                color: #e53e3e; 
                background: #fed7d7; 
                padding: 12px; 
                border-radius: 8px; 
                margin-top: 15px; 
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h1>AMAC Therapy</h1>
            <p class="subtitle">Speech Improvement System</p>
            
            <div class="error" id="errorMessage"></div>
            
            <input type="text" id="username" placeholder="Username" value="testuser">
            <input type="password" id="password" placeholder="Password" value="amac2024">
            <button onclick="login()">Login to Dashboard</button>
            
            <div class="links">
                <p>Don't have an account? <a href="/signup">Sign up here</a></p>
                <p><a href="/therapy">Continue as Guest</a> | <a href="#" onclick="clearStorage()">Clear Login</a></p>
            </div>
            
            <div class="demo">
                <strong>Demo Credentials:</strong><br>
                Username: <strong>testuser</strong><br>
                Password: <strong>amac2024</strong>
            </div>
        </div>
        
        <script>
        function showError(message) {
            const errorDiv = document.getElementById('errorMessage');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
        
        function clearError() {
            document.getElementById('errorMessage').style.display = 'none';
        }
        
        function login() {
            const user = document.getElementById('username').value;
            const pass = document.getElementById('password').value;
            
            if (!user || !pass) {
                showError('Please enter both username and password');
                return;
            }
            
            // Simple validation - replace with actual API call in production
            if (user === 'testuser' && pass === 'amac2024') {
                localStorage.setItem('amacLoggedIn', 'true');
                localStorage.setItem('amacUsername', user);
                window.location.href = '/therapy';
            } else {
                showError('Invalid credentials. Use: testuser / amac2024');
            }
        }
        
        function clearStorage() {
            localStorage.removeItem('amacLoggedIn');
            localStorage.removeItem('amacUsername');
            alert('Local storage cleared. Refresh the page.');
        }
        
        // Enter key support
        document.getElementById('password').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                clearError();
                login();
            }
        });
        
        // Optional: Clear error when typing
        document.getElementById('username').addEventListener('input', clearError);
        document.getElementById('password').addEventListener('input', clearError);
        
        // Check if already logged in - but don't auto-redirect immediately
        // Let users see the login page first
        window.onload = function() {
            const isLoggedIn = localStorage.getItem('amacLoggedIn');
            if (isLoggedIn) {
                // Show a message that they're already logged in
                document.getElementById('username').value = localStorage.getItem('amacUsername') || '';
                document.getElementById('password').value = '******';
                showError('You are already logged in. Click "Login to Dashboard" to continue or "Clear Login" to start fresh.');
            }
        }
        </script>
    </body>
    </html>
    """)
@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/api/login")
async def api_login(request: Request):
    data = await request.json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return JSONResponse({"success": False, "message": "Username and password required"})
    
    conn = sqlite3.connect('amac_users.db')
    c = conn.cursor()
    
    # Check if user exists
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result and result[0] == hash_password(password):
        return JSONResponse({
            "success": True, 
            "message": "Login successful",
            "username": username,
            "redirect": "/therapy"
        })
    else:
        return JSONResponse({"success": False, "message": "Invalid username or password"})


# Signup page
@app.get("/signup")
async def signup_page():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AMAC Therapy - Sign Up</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                min-height: 100vh; 
                padding: 20px; 
            }
            .signup-box { 
                background: white; 
                padding: 35px; 
                border-radius: 15px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.2); 
                width: 420px; 
            }
            h1 { 
                color: #2c5aa0; 
                text-align: center; 
                margin-bottom: 8px; 
                font-size: 28px; 
            }
            .subtitle { 
                color: #666; 
                text-align: center; 
                margin-bottom: 25px; 
                font-size: 16px; 
            }
            input { 
                width: 100%; 
                padding: 14px; 
                margin: 8px 0; 
                border: 2px solid #e1e5e9; 
                border-radius: 8px; 
                box-sizing: border-box; 
                font-size: 16px; 
                transition: border 0.3s;
            }
            input:focus { 
                outline: none; 
                border-color: #2c5aa0; 
                box-shadow: 0 0 0 3px rgba(44,90,160,0.1);
            }
            .row { 
                display: flex; 
                gap: 15px; 
            }
            .row input { flex: 1; }
            button { 
                width: 100%; 
                background: linear-gradient(135deg, #2c5aa0 0%, #1e3c72 100%); 
                color: white; 
                border: none; 
                padding: 16px; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 16px; 
                font-weight: 600; 
                margin-top: 15px; 
                transition: transform 0.2s;
            }
            button:hover { 
                transform: translateY(-2px); 
                box-shadow: 0 5px 15px rgba(44,90,160,0.3);
            }
            .terms { 
                font-size: 14px; 
                margin: 20px 0; 
                color: #666;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .login-link { 
                text-align: center; 
                margin-top: 25px; 
                font-size: 14px; 
                color: #666;
            }
            a { 
                color: #2c5aa0; 
                text-decoration: none; 
                font-weight: 500;
            }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="signup-box">
            <h1>AMAC Therapy</h1>
            <p class="subtitle">Speech Improvement System</p>
            <div class="row">
                <input type="text" placeholder="First Name">
                <input type="text" placeholder="Last Name">
            </div>
            <input type="email" placeholder="Email Address">
            <input type="text" placeholder="Username">
            <input type="password" placeholder="Password">
            <input type="password" placeholder="Confirm Password">
            <div class="terms">
                <input type="checkbox" id="terms" style="width: auto;"> 
                <label for="terms">I agree to Terms & Privacy Policy</label>
            </div>
            <button onclick="signup()">Create Account</button>
            <div class="login-link">
                Already have an account? <a href="/login">Login here</a>
            </div>
        </div>
        <script>
        function signup() {
            alert('Account created successfully! Redirecting to login...');
            window.location.href = '/login';
        }
        </script>
    </body>
    </html>
    """)
@app.get("/signup")
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/api/signup")
async def api_signup(request: Request):
    data = await request.json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not all([username, email, password]):
        return JSONResponse({"success": False, "message": "All fields are required"})
    
    if len(password) < 6:
        return JSONResponse({"success": False, "message": "Password must be at least 6 characters"})
    
    conn = sqlite3.connect('amac_users.db')
    c = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        c.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        conn.commit()
        return JSONResponse({"success": True, "message": "Account created successfully! Please login."})
    except sqlite3.IntegrityError as e:
        error_msg = "Username or email already exists"
        if "email" in str(e).lower():
            error_msg = "Email already registered"
        elif "username" in str(e).lower():
            error_msg = "Username already taken"
        return JSONResponse({"success": False, "message": error_msg})
    finally:
        conn.close()

# Therapy page (main dashboard)
@app.get("/therapy")
async def therapy_page():
    # In production, add authentication here
    return FileResponse("app/static/therapy.html")

# ============ API ENDPOINTS ============

@app.get("/api/health")
def health():
    return {"status": "OK", "service": "AMAC Therapy"}

@app.get("/api/user/profile")
def profile():
    return {
        "name": "Test User",
        "impairment_level": "moderate",
        "total_sessions": 5,
        "day_streak": 3,
        "overall_score": 72,
        "member_since": "Jan 2024",
        "current_level": "Intermediate",
        "next_milestone": "10 Sessions"
    }

@app.post("/api/therapy/start")
def start_session():
    return {
        "session_id": "test_session_123",
        "message": "Session started",
        "first_exercise": {
            "exercise_number": 1,
            "total_exercises": 4,
            "target": "AH",
            "instructions": "Say 'AH' with open mouth",
            "type": "phoneme",
            "difficulty": 1
        }
    }

@app.get("/api/dashboard/stats")
def dashboard_stats():
    return {
        "total_sessions": 5,
        "best_score": 85,
        "day_streak": 3,
        "avg_score": 72,
        "recent_activity": [
            {
                "session": "Today",
                "exercises_completed": 4,
                "avg_score": 78
            },
            {
                "session": "Yesterday",
                "exercises_completed": 3,
                "avg_score": 72
            }
        ]
    }

# ============ MAIN ============

if __name__ == "__main__":
    import uvicorn
    print("🚀" * 30)
    print("AMAC THERAPY SERVER STARTING")
    print("🚀" * 30)
    print("\n📌 ACCESS LINKS:")
    print("   Login Page:     http://localhost:8000/login")
    print("   Signup Page:    http://localhost:8000/signup")
    print("   Therapy Page:   http://localhost:8000/therapy")
    print("   API Health:     http://localhost:8000/api/health")
    print("   Static Files:   http://localhost:8000/static/")
    print("\n🔑 DEMO CREDENTIALS:")
    print("   Username: testuser")
    print("   Password: amac2024")
    print("\n" + "=" * 50)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"\n⚠️  Port 8000 is busy! Trying port 8080...")
        print(f"\n📌 ACCESS LINKS (Port 8080):")
        print("   Login Page:     http://localhost:8080/login")
        print("   Signup Page:    http://localhost:8080/signup")
        print("   Therapy Page:   http://localhost:8080/therapy")
        uvicorn.run(app, host="0.0.0.0", port=8080)