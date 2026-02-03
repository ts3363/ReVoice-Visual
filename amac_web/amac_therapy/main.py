from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta
import sqlite3
import hashlib
import os

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
    
    # Create user_progress table for tracking per-user stats
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            total_sessions INTEGER DEFAULT 0,
            total_exercises INTEGER DEFAULT 0,
            best_score INTEGER DEFAULT 0,
            avg_score REAL DEFAULT 0,
            day_streak INTEGER DEFAULT 0,
            last_session_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    
    # Create session_history table for detailed session tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS session_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            session_date TEXT,
            exercises_completed INTEGER DEFAULT 0,
            avg_score REAL DEFAULT 0,
            duration_minutes INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    
    conn.commit()
    
    # Add default test user if not exists
    try:
        test_password_hash = hashlib.sha256('amac2024'.encode()).hexdigest()
        c.execute(
            "INSERT OR IGNORE INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ('testuser', 'test@amac.com', test_password_hash)
        )
        conn.commit()
    except Exception as e:
        print(f"Note: Default user may already exist: {e}")
    
    conn.close()

# Initialize database on startup
init_db()

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ============ WEB PAGES ============

# Root - redirect to login
@app.get("/")
async def root():
    return RedirectResponse(url="/login")

# Login page
@app.get("/login")
async def login_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AMAC Therapy - Login</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
            display: flex;
        }
        
        .left-panel {
            flex: 1;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 60px;
            position: relative;
            overflow: hidden;
        }
        
        .left-panel::before {
            content: '';
            position: absolute;
            width: 400px;
            height: 400px;
            background: rgba(255,255,255,0.1);
            border-radius: 50%;
            top: -100px;
            left: -100px;
            animation: float 6s ease-in-out infinite;
        }
        
        .left-panel::after {
            content: '';
            position: absolute;
            width: 300px;
            height: 300px;
            background: rgba(255,255,255,0.08);
            border-radius: 50%;
            bottom: -50px;
            right: -50px;
            animation: float 8s ease-in-out infinite reverse;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
        }
        
        .brand {
            text-align: center;
            color: white;
            z-index: 1;
        }
        
        .brand-icon {
            width: 100px;
            height: 100px;
            background: rgba(255,255,255,0.2);
            border-radius: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 30px;
            font-size: 45px;
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .brand h1 {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 15px;
        }
        
        .brand p {
            font-size: 18px;
            opacity: 0.9;
            max-width: 350px;
            line-height: 1.6;
        }
        
        .features {
            margin-top: 50px;
            z-index: 1;
        }
        
        .feature {
            display: flex;
            align-items: center;
            gap: 15px;
            color: white;
            margin: 20px 0;
            opacity: 0.95;
        }
        
        .feature i {
            width: 40px;
            height: 40px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .right-panel {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 40px;
            background: #f8fafc;
        }
        
        .login-card {
            width: 100%;
            max-width: 420px;
            background: white;
            padding: 50px 40px;
            border-radius: 24px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.08);
        }
        
        .login-card h2 {
            font-size: 28px;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 8px;
        }
        
        .login-card .subtitle {
            color: #6b7280;
            margin-bottom: 35px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        .input-group label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: #374151;
            margin-bottom: 8px;
        }
        
        .input-wrapper {
            position: relative;
            display: flex;
            align-items: center;
        }
        
        .input-wrapper > i:first-child {
            position: absolute;
            left: 16px;
            top: 50%;
            transform: translateY(-50%);
            color: #9ca3af;
            z-index: 1;
            pointer-events: none;
        }
        
        .input-wrapper input {
            width: 100%;
            padding: 14px 50px 14px 48px;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            font-size: 15px;
            font-family: inherit;
            transition: all 0.3s;
            background: white;
        }
        
        .input-wrapper input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        }
        
        .toggle-password {
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            background: none;
            border: none;
            color: #9ca3af;
            cursor: pointer;
            padding: 5px;
            z-index: 2;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .toggle-password:hover {
            color: #667eea;
        }
        
        .toggle-password i {
            position: static;
            transform: none;
        }
        
        .remember-forgot {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
        }
        
        .remember-forgot label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #6b7280;
            cursor: pointer;
        }
        
        .remember-forgot a {
            font-size: 14px;
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }
        
        .btn-login {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        
        .btn-login:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        
        .btn-login:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-login .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .divider {
            display: flex;
            align-items: center;
            margin: 25px 0;
            color: #9ca3af;
            font-size: 14px;
        }
        
        .divider::before,
        .divider::after {
            content: '';
            flex: 1;
            height: 1px;
            background: #e5e7eb;
        }
        
        .divider span {
            padding: 0 15px;
        }
        
        .signup-link {
            text-align: center;
            margin-top: 25px;
            color: #6b7280;
            font-size: 15px;
        }
        
        .signup-link a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        
        .signup-link a:hover {
            text-decoration: underline;
        }
        
        .error-message {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #dc2626;
            padding: 12px 16px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            display: none;
            align-items: center;
            gap: 10px;
        }
        
        .error-message.show {
            display: flex;
        }
        
        @media (max-width: 900px) {
            body { flex-direction: column; }
            .left-panel { padding: 40px 30px; min-height: 40vh; }
            .brand h1 { font-size: 32px; }
            .features { display: none; }
            .right-panel { padding: 30px 20px; }
            .login-card { padding: 35px 25px; }
        }
    </style>
</head>
<body>
    <div class="left-panel">
        <div class="brand">
            <div class="brand-icon">
                <i class="fas fa-comment-medical"></i>
            </div>
            <h1>AMAC Therapy</h1>
            <p>Advanced speech improvement system powered by AI technology</p>
        </div>
        
        <div class="features">
            <div class="feature">
                <i class="fas fa-microphone-alt"></i>
                <span>Real-time speech analysis</span>
            </div>
            <div class="feature">
                <i class="fas fa-chart-line"></i>
                <span>Track your progress over time</span>
            </div>
            <div class="feature">
                <i class="fas fa-brain"></i>
                <span>Personalized therapy exercises</span>
            </div>
        </div>
    </div>
    
    <div class="right-panel">
        <div class="login-card">
            <h2>Welcome back!</h2>
            <p class="subtitle">Sign in to continue your therapy journey</p>
            
            <div class="error-message" id="errorMsg">
                <i class="fas fa-exclamation-circle"></i>
                <span id="errorText"></span>
            </div>
            
            <form id="loginForm" onsubmit="return handleLogin(event)">
                <div class="input-group">
                    <label>Username</label>
                    <div class="input-wrapper">
                        <i class="fas fa-user"></i>
                        <input type="text" id="username" placeholder="Enter your username" required>
                    </div>
                </div>
                
                <div class="input-group">
                    <label>Password</label>
                    <div class="input-wrapper">
                        <i class="fas fa-lock"></i>
                        <input type="password" id="password" placeholder="Enter your password" required>
                        <button type="button" class="toggle-password" onclick="togglePassword()">
                            <i class="fas fa-eye" id="eyeIcon"></i>
                        </button>
                    </div>
                </div>
                
                <div class="remember-forgot">
                    <label>
                        <input type="checkbox" id="remember"> Remember me
                    </label>
                    <a href="#">Forgot password?</a>
                </div>
                
                <button type="submit" class="btn-login" id="loginBtn">
                    <span class="spinner" id="spinner"></span>
                    <span id="btnText">Sign In</span>
                </button>
            </form>
            
            <div class="signup-link">
                Don't have an account? <a href="/signup">Create one</a>
            </div>
        </div>
    </div>
    
    <script>
        function togglePassword() {
            const pwd = document.getElementById('password');
            const icon = document.getElementById('eyeIcon');
            if (pwd.type === 'password') {
                pwd.type = 'text';
                icon.className = 'fas fa-eye-slash';
            } else {
                pwd.type = 'password';
                icon.className = 'fas fa-eye';
            }
        }
        
        function showError(msg) {
            document.getElementById('errorText').textContent = msg;
            document.getElementById('errorMsg').classList.add('show');
        }
        
        function hideError() {
            document.getElementById('errorMsg').classList.remove('show');
        }
        
        async function handleLogin(e) {
            e.preventDefault();
            hideError();
            
            const user = document.getElementById('username').value.trim();
            const pass = document.getElementById('password').value;
            const btn = document.getElementById('loginBtn');
            const spinner = document.getElementById('spinner');
            const btnText = document.getElementById('btnText');
            
            if (!user || !pass) {
                showError('Please enter username and password');
                return false;
            }
            
            // Show loading
            btn.disabled = true;
            spinner.style.display = 'inline-block';
            btnText.textContent = 'Signing in...';
            
            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        username: user,
                        password: pass
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    localStorage.setItem('amacLoggedIn', 'true');
                    localStorage.setItem('amacUsername', data.username);
                    window.location.href = data.redirect || '/therapy';
                } else {
                    showError(data.message || 'Invalid credentials');
                    btn.disabled = false;
                    spinner.style.display = 'none';
                    btnText.textContent = 'Sign In';
                }
            } catch (error) {
                console.error('Login error:', error);
                showError('Connection error. Please try again.');
                btn.disabled = false;
                spinner.style.display = 'none';
                btnText.textContent = 'Sign In';
            }
            
            return false;
        }
        
        // Check if already logged in
        if (localStorage.getItem('amacLoggedIn') === 'true') {
            document.getElementById('username').value = localStorage.getItem('amacUsername') || '';
        }
        
        // Input focus effects
        document.querySelectorAll('input').forEach(input => {
            input.addEventListener('focus', hideError);
        });
    </script>
</body>
</html>
    """)

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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AMAC Therapy - Sign Up</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
            display: flex;
        }
        
        .left-panel {
            flex: 1;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 60px;
            position: relative;
            overflow: hidden;
        }
        
        .left-panel::before {
            content: '';
            position: absolute;
            width: 400px;
            height: 400px;
            background: rgba(255,255,255,0.1);
            border-radius: 50%;
            top: -100px;
            left: -100px;
            animation: float 6s ease-in-out infinite;
        }
        
        .left-panel::after {
            content: '';
            position: absolute;
            width: 300px;
            height: 300px;
            background: rgba(255,255,255,0.08);
            border-radius: 50%;
            bottom: -50px;
            right: -50px;
            animation: float 8s ease-in-out infinite reverse;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
        }
        
        .brand {
            text-align: center;
            color: white;
            z-index: 1;
        }
        
        .brand-icon {
            width: 100px;
            height: 100px;
            background: rgba(255,255,255,0.2);
            border-radius: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 30px;
            font-size: 45px;
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .brand h1 {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 15px;
        }
        
        .brand p {
            font-size: 18px;
            opacity: 0.9;
            max-width: 350px;
            line-height: 1.6;
        }
        
        .steps {
            margin-top: 50px;
            z-index: 1;
        }
        
        .step {
            display: flex;
            align-items: center;
            gap: 15px;
            color: white;
            margin: 20px 0;
            opacity: 0.95;
        }
        
        .step-num {
            width: 40px;
            height: 40px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
        }
        
        .right-panel {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 40px;
            background: #f8fafc;
        }
        
        .signup-card {
            width: 100%;
            max-width: 480px;
            background: white;
            padding: 45px 40px;
            border-radius: 24px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.08);
        }
        
        .signup-card h2 {
            font-size: 28px;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 8px;
        }
        
        .signup-card .subtitle {
            color: #6b7280;
            margin-bottom: 30px;
        }
        
        .row {
            display: flex;
            gap: 15px;
        }
        
        .row .input-group {
            flex: 1;
        }
        
        .input-group {
            margin-bottom: 18px;
        }
        
        .input-group label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: #374151;
            margin-bottom: 8px;
        }
        
        .input-wrapper {
            position: relative;
        }
        
        .input-wrapper i {
            position: absolute;
            left: 16px;
            top: 50%;
            transform: translateY(-50%);
            color: #9ca3af;
        }
        
        .input-wrapper input {
            width: 100%;
            padding: 14px 16px 14px 48px;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            font-size: 15px;
            font-family: inherit;
            transition: all 0.3s;
        }
        
        .input-wrapper input:focus {
            outline: none;
            border-color: #10b981;
            box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.1);
        }
        
        .input-wrapper.success input {
            border-color: #10b981;
        }
        
        .input-wrapper.error input {
            border-color: #ef4444;
        }
        
        .password-strength {
            margin-top: 8px;
            height: 4px;
            background: #e5e7eb;
            border-radius: 2px;
            overflow: hidden;
        }
        
        .password-strength .bar {
            height: 100%;
            width: 0;
            transition: all 0.3s;
        }
        
        .strength-text {
            font-size: 12px;
            margin-top: 5px;
            color: #6b7280;
        }
        
        .terms {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin: 25px 0;
            font-size: 14px;
            color: #6b7280;
        }
        
        .terms input {
            margin-top: 3px;
        }
        
        .terms a {
            color: #10b981;
            text-decoration: none;
        }
        
        .btn-signup {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-signup:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.4);
        }
        
        .btn-signup:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }
        
        .login-link {
            text-align: center;
            margin-top: 25px;
            color: #6b7280;
            font-size: 15px;
        }
        
        .login-link a {
            color: #10b981;
            text-decoration: none;
            font-weight: 600;
        }
        
        .error-message {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #dc2626;
            padding: 12px 16px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            display: none;
            align-items: center;
            gap: 10px;
        }
        
        .error-message.show {
            display: flex;
        }
        
        .success-message {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #16a34a;
            padding: 12px 16px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            display: none;
            align-items: center;
            gap: 10px;
        }
        
        .success-message.show {
            display: flex;
        }
        
        @media (max-width: 900px) {
            body { flex-direction: column; }
            .left-panel { padding: 40px 30px; min-height: 35vh; }
            .brand h1 { font-size: 32px; }
            .steps { display: none; }
            .right-panel { padding: 30px 20px; }
            .signup-card { padding: 35px 25px; }
            .row { flex-direction: column; gap: 0; }
        }
    </style>
</head>
<body>
    <div class="left-panel">
        <div class="brand">
            <div class="brand-icon">
                <i class="fas fa-user-plus"></i>
            </div>
            <h1>Join AMAC</h1>
            <p>Start your speech therapy journey today with personalized exercises</p>
        </div>
        
        <div class="steps">
            <div class="step">
                <div class="step-num">1</div>
                <span>Create your account</span>
            </div>
            <div class="step">
                <div class="step-num">2</div>
                <span>Complete assessment</span>
            </div>
            <div class="step">
                <div class="step-num">3</div>
                <span>Start therapy sessions</span>
            </div>
        </div>
    </div>
    
    <div class="right-panel">
        <div class="signup-card">
            <h2>Create Account</h2>
            <p class="subtitle">Fill in your details to get started</p>
            
            <div class="error-message" id="errorMsg">
                <i class="fas fa-exclamation-circle"></i>
                <span id="errorText"></span>
            </div>
            
            <div class="success-message" id="successMsg">
                <i class="fas fa-check-circle"></i>
                <span id="successText"></span>
            </div>
            
            <form id="signupForm" onsubmit="return handleSignup(event)">
                <div class="row">
                    <div class="input-group">
                        <label>First Name</label>
                        <div class="input-wrapper">
                            <i class="fas fa-user"></i>
                            <input type="text" id="firstName" placeholder="John" required>
                        </div>
                    </div>
                    <div class="input-group">
                        <label>Last Name</label>
                        <div class="input-wrapper">
                            <i class="fas fa-user"></i>
                            <input type="text" id="lastName" placeholder="Doe" required>
                        </div>
                    </div>
                </div>
                
                <div class="input-group">
                    <label>Email Address</label>
                    <div class="input-wrapper" id="emailWrapper">
                        <i class="fas fa-envelope"></i>
                        <input type="email" id="email" placeholder="john@example.com" required>
                    </div>
                </div>
                
                <div class="input-group">
                    <label>Username</label>
                    <div class="input-wrapper">
                        <i class="fas fa-at"></i>
                        <input type="text" id="username" placeholder="johndoe" required>
                    </div>
                </div>
                
                <div class="input-group">
                    <label>Password</label>
                    <div class="input-wrapper">
                        <i class="fas fa-lock"></i>
                        <input type="password" id="password" placeholder="Min 6 characters" required oninput="checkStrength()">
                    </div>
                    <div class="password-strength">
                        <div class="bar" id="strengthBar"></div>
                    </div>
                    <div class="strength-text" id="strengthText"></div>
                </div>
                
                <div class="input-group">
                    <label>Confirm Password</label>
                    <div class="input-wrapper" id="confirmWrapper">
                        <i class="fas fa-lock"></i>
                        <input type="password" id="confirmPassword" placeholder="Confirm password" required oninput="checkMatch()">
                    </div>
                </div>
                
                <div class="terms">
                    <input type="checkbox" id="terms" required>
                    <label for="terms">I agree to the <a href="#">Terms of Service</a> and <a href="#">Privacy Policy</a></label>
                </div>
                
                <button type="submit" class="btn-signup" id="signupBtn">Create Account</button>
            </form>
            
            <div class="login-link">
                Already have an account? <a href="/login">Sign in</a>
            </div>
        </div>
    </div>
    
    <script>
        function checkStrength() {
            const pwd = document.getElementById('password').value;
            const bar = document.getElementById('strengthBar');
            const text = document.getElementById('strengthText');
            
            let strength = 0;
            if (pwd.length >= 6) strength++;
            if (pwd.length >= 8) strength++;
            if (/[A-Z]/.test(pwd)) strength++;
            if (/[0-9]/.test(pwd)) strength++;
            if (/[^A-Za-z0-9]/.test(pwd)) strength++;
            
            const colors = ['#ef4444', '#f59e0b', '#eab308', '#84cc16', '#10b981'];
            const labels = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong'];
            
            const idx = Math.min(strength, 4);
            bar.style.width = ((strength + 1) * 20) + '%';
            bar.style.background = colors[idx];
            text.textContent = strength > 0 ? labels[idx] : '';
            text.style.color = colors[idx];
        }
        
        function checkMatch() {
            const pwd = document.getElementById('password').value;
            const confirm = document.getElementById('confirmPassword').value;
            const wrapper = document.getElementById('confirmWrapper');
            
            if (confirm.length > 0) {
                if (pwd === confirm) {
                    wrapper.className = 'input-wrapper success';
                } else {
                    wrapper.className = 'input-wrapper error';
                }
            } else {
                wrapper.className = 'input-wrapper';
            }
        }
        
        function showError(msg) {
            document.getElementById('errorText').textContent = msg;
            document.getElementById('errorMsg').classList.add('show');
            document.getElementById('successMsg').classList.remove('show');
        }
        
        function showSuccess(msg) {
            document.getElementById('successText').textContent = msg;
            document.getElementById('successMsg').classList.add('show');
            document.getElementById('errorMsg').classList.remove('show');
        }
        
        async function handleSignup(e) {
            e.preventDefault();
            
            const user = document.getElementById('username').value.trim();
            const email = document.getElementById('email').value.trim();
            const pwd = document.getElementById('password').value;
            const confirm = document.getElementById('confirmPassword').value;
            
            if (!user || !email || !pwd) {
                showError('All fields are required');
                return false;
            }
            
            if (pwd.length < 6) {
                showError('Password must be at least 6 characters');
                return false;
            }
            
            if (pwd !== confirm) {
                showError('Passwords do not match');
                return false;
            }
            
            if (!document.getElementById('terms').checked) {
                showError('Please agree to the terms');
                return false;
            }
            
            // Call the API to create account
            const btn = document.querySelector('.btn-signup');
            btn.disabled = true;
            btn.textContent = 'Creating account...';
            
            try {
                const response = await fetch('/api/signup', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        username: user,
                        email: email,
                        password: pwd
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showSuccess('Account created successfully! Redirecting to login...');
                    setTimeout(() => {
                        window.location.href = '/login';
                    }, 2000);
                } else {
                    showError(data.message || 'Failed to create account');
                    btn.disabled = false;
                    btn.textContent = 'Create Account';
                }
            } catch (error) {
                console.error('Signup error:', error);
                showError('Connection error. Please try again.');
                btn.disabled = false;
                btn.textContent = 'Create Account';
            }
            
            return false;
        }
    </script>
</body>
</html>
    """)

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
    from fastapi.responses import HTMLResponse
    import os
    file_path = "app/static/therapy.html"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

# Feedback page
@app.get("/feedback")
async def feedback_page():
    from fastapi.responses import HTMLResponse
    file_path = "app/static/feedback.html"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

# Help page
@app.get("/help")
async def help_page():
    from fastapi.responses import HTMLResponse
    file_path = "app/static/help.html"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

# Logout - clear session and redirect to login
@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    # Clear any session cookies
    response.delete_cookie("session")
    response.delete_cookie("user")
    response.delete_cookie("username")
    return response

# ============ API ENDPOINTS ============

@app.get("/api/health")
def health():
    return {"status": "OK", "service": "AMAC Therapy"}

@app.get("/api/user/profile")
def profile(username: str = None):
    # Get user from database if username provided
    if username:
        conn = sqlite3.connect('amac_users.db')
        c = conn.cursor()
        c.execute("SELECT username, email, created_at FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        
        if result:
            # Check if user has progress data
            c.execute("SELECT total_sessions, best_score, avg_score, day_streak FROM user_progress WHERE username = ?", (username,))
            progress = c.fetchone()
            conn.close()
            
            if progress:
                # Return existing user progress
                return {
                    "name": result[0],
                    "email": result[1] if result[1] else "",
                    "impairment_level": "moderate",
                    "total_sessions": progress[0],
                    "day_streak": progress[3],
                    "overall_score": int(progress[2]) if progress[2] else 0,
                    "member_since": result[2][:10] if result[2] else "New",
                    "current_level": "Beginner" if progress[0] < 5 else ("Intermediate" if progress[0] < 15 else "Advanced"),
                    "next_milestone": f"{5 - progress[0]} more sessions" if progress[0] < 5 else f"{15 - progress[0]} more sessions"
                }
            else:
                # Fresh user - no progress yet
                return {
                    "name": result[0],
                    "email": result[1] if result[1] else "",
                    "impairment_level": "moderate",
                    "total_sessions": 0,
                    "day_streak": 0,
                    "overall_score": 0,
                    "member_since": result[2][:10] if result[2] else "New",
                    "current_level": "Beginner",
                    "next_milestone": "Complete first session"
                }
        conn.close()
    
    # Default fallback for guest
    return {
        "name": "Guest User",
        "impairment_level": "moderate",
        "total_sessions": 0,
        "day_streak": 0,
        "overall_score": 0,
        "member_since": "New",
        "current_level": "Beginner",
        "next_milestone": "First Session"
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
def dashboard_stats(username: str = None):
    if username:
        conn = sqlite3.connect('amac_users.db')
        c = conn.cursor()
        
        # Get user progress
        c.execute("SELECT total_sessions, best_score, avg_score, day_streak FROM user_progress WHERE username = ?", (username,))
        progress = c.fetchone()
        
        # Get recent sessions
        c.execute("SELECT session_date, exercises_completed, avg_score FROM session_history WHERE username = ? ORDER BY created_at DESC LIMIT 5", (username,))
        sessions = c.fetchall()
        conn.close()
        
        if progress:
            recent_activity = []
            for session in sessions:
                recent_activity.append({
                    "session": session[0],
                    "exercises_completed": session[1],
                    "avg_score": int(session[2]) if session[2] else 0
                })
            
            return {
                "total_sessions": progress[0],
                "best_score": progress[1],
                "day_streak": progress[3],
                "avg_score": int(progress[2]) if progress[2] else 0,
                "recent_activity": recent_activity
            }
    
    # Fresh user - no data yet
    return {
        "total_sessions": 0,
        "best_score": 0,
        "day_streak": 0,
        "avg_score": 0,
        "recent_activity": []
    }

# ============ MISSING API ENDPOINTS ============

@app.get("/api/user/progress")
async def get_user_progress(username: str = "default"):
    """Get user's progress data"""
    conn = sqlite3.connect('amac_users.db')
    c = conn.cursor()
    
    # Get user progress
    c.execute("SELECT total_sessions, total_exercises, best_score, avg_score, day_streak FROM user_progress WHERE username = ?", (username,))
    progress = c.fetchone()
    
    # Get recent session scores in chronological order (oldest first for chart)
    c.execute("SELECT avg_score, session_date FROM session_history WHERE username = ? ORDER BY created_at ASC LIMIT 10", (username,))
    session_rows = c.fetchall()
    recent_scores = [int(row[0]) if row[0] else 0 for row in session_rows]
    session_dates = [row[1] for row in session_rows]
    
    conn.close()
    
    if progress and progress[0] > 0:  # Has sessions
        return {
            "username": username,
            "best_score": progress[2],
            "total_exercises": progress[1],
            "total_sessions": progress[0],
            "avg_score": int(progress[3]) if progress[3] else 0,
            "average_clarity": int(progress[3]) if progress[3] else 0,
            "day_streak": progress[4],
            "weekly_progress": recent_scores if recent_scores else [0],
            "session_dates": session_dates if session_dates else ["Today"],
            "monthly_sessions": [progress[0]],
            "achievements": [
                {"name": "First Session", "earned": progress[0] >= 1},
                {"name": "5 Day Streak", "earned": progress[4] >= 5},
                {"name": "Score 90+", "earned": progress[2] >= 90}
            ],
            "recent_scores": recent_scores if recent_scores else [0],
            "improvement_rate": 0
        }
    
    # Fresh user - no progress yet
    return {
        "username": username,
        "best_score": 0,
        "total_exercises": 0,
        "total_sessions": 0,
        "avg_score": 0,
        "average_clarity": 0,
        "day_streak": 0,
        "weekly_progress": [0],
        "monthly_sessions": [0],
        "achievements": [
            {"name": "First Session", "earned": False},
            {"name": "5 Day Streak", "earned": False},
            {"name": "Score 90+", "earned": False}
        ],
        "recent_scores": [0],
        "improvement_rate": 0
    }

@app.get("/api/therapy/sessions")
async def get_therapy_sessions(user_id: str = None, username: str = None):
    """Get user's therapy session history"""
    # Support both user_id and username parameters
    target_user = username or user_id or "default"
    
    conn = sqlite3.connect('amac_users.db')
    c = conn.cursor()
    
    # Get user sessions in chronological order (oldest first for charts)
    c.execute("SELECT id, session_date, exercises_completed, avg_score, duration_minutes FROM session_history WHERE username = ? ORDER BY created_at ASC LIMIT 20", (target_user,))
    rows = c.fetchall()
    conn.close()
    
    sessions = []
    for row in rows:
        sessions.append({
            "session_id": f"session_{row[0]:03d}",
            "date": row[1],
            "exercises_completed": row[2],
            "avg_score": int(row[3]) if row[3] else 0,
            "score": int(row[3]) if row[3] else 0,  # Alias for charts
            "duration_minutes": row[4],
            "status": "completed"
        })
    
    return {
        "user_id": target_user,
        "total_sessions": len(sessions),
        "sessions": sessions
    }

@app.post("/api/therapy/process-attempt")
async def process_therapy_attempt(request: Request):
    """Process a speech therapy attempt and return feedback"""
    try:
        data = await request.json()
        target = data.get('target', '')
        exercise_type = data.get('type', 'phoneme')
        audio_data = data.get('audio_data', None)
        
        # Simulate speech analysis (in production, this would use actual ML models)
        import random
        
        # Generate realistic score based on difficulty
        base_score = random.randint(65, 95)
        
        # Generate feedback based on score
        if base_score >= 85:
            feedback = "Excellent! Your pronunciation was very clear."
            suggestions = ["Keep up the great work!", "Try increasing your speaking speed slightly."]
            rating = "excellent"
        elif base_score >= 70:
            feedback = "Good job! There's room for improvement."
            suggestions = ["Focus on breath support.", "Try to elongate vowel sounds."]
            rating = "good"
        else:
            feedback = "Nice try! Let's work on this together."
            suggestions = ["Slow down your speech.", "Practice the sound in isolation first.", "Focus on mouth positioning."]
            rating = "needs_practice"
        
        return {
            "success": True,
            "score": base_score,
            "feedback": feedback,
            "rating": rating,
            "suggestions": suggestions,
            "details": {
                "clarity": random.randint(60, 100),
                "accuracy": random.randint(60, 100),
                "fluency": random.randint(60, 100),
                "volume": random.randint(70, 100)
            },
            "target": target,
            "exercise_type": exercise_type
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/therapy/save-progress")
async def save_therapy_progress(request: Request):
    """Save user's therapy progress"""
    try:
        data = await request.json()
        username = data.get('username', 'default')
        exercises = data.get('exercises', [])
        session_score = data.get('session_score', 0)
        duration = data.get('duration_minutes', 0)
        
        if not exercises:
            return {"success": True, "message": "No exercises to save"}
        
        # Calculate average score for this session
        total_score = sum(ex.get('score', 0) for ex in exercises)
        avg_score = total_score / len(exercises) if exercises else 0
        
        conn = sqlite3.connect('amac_users.db')
        c = conn.cursor()
        
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Add to session history
        c.execute('''
            INSERT INTO session_history (username, session_date, exercises_completed, avg_score, duration_minutes)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, today, len(exercises), avg_score, duration))
        
        # Check if user has progress record
        c.execute("SELECT total_sessions, total_exercises, best_score, avg_score, day_streak, last_session_date FROM user_progress WHERE username = ?", (username,))
        progress = c.fetchone()
        
        if progress:
            # Update existing progress
            new_total_sessions = progress[0] + 1
            new_total_exercises = progress[1] + len(exercises)
            new_best_score = max(progress[2], int(avg_score))
            # Recalculate running average
            new_avg_score = ((progress[3] * progress[0]) + avg_score) / new_total_sessions if new_total_sessions > 0 else avg_score
            
            # Check streak - if last session was yesterday, increment; if today, keep; else reset to 1
            last_date = progress[5]
            if last_date == today:
                new_streak = progress[4]  # Same day, keep streak
            elif last_date == (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"):
                new_streak = progress[4] + 1  # Yesterday, increment
            else:
                new_streak = 1  # Gap, reset streak
            
            c.execute('''
                UPDATE user_progress 
                SET total_sessions = ?, total_exercises = ?, best_score = ?, avg_score = ?, 
                    day_streak = ?, last_session_date = ?, updated_at = CURRENT_TIMESTAMP
                WHERE username = ?
            ''', (new_total_sessions, new_total_exercises, new_best_score, new_avg_score, new_streak, today, username))
        else:
            # Create new progress record
            c.execute('''
                INSERT INTO user_progress (username, total_sessions, total_exercises, best_score, avg_score, day_streak, last_session_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, 1, len(exercises), int(avg_score), avg_score, 1, today))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "Progress saved successfully",
            "exercises_saved": len(exercises),
            "username": username,
            "session_score": int(avg_score)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/therapy/exercise")
async def get_next_exercise(session_id: str = None, current: int = 1):
    """Get next therapy exercise"""
    exercises = [
        {"target": "AH", "instructions": "Say 'AH' with open mouth, hold for 3 seconds", "type": "phoneme", "difficulty": 1},
        {"target": "EE", "instructions": "Smile wide, say 'EE' clearly", "type": "phoneme", "difficulty": 1},
        {"target": "Hello", "instructions": "Say 'Hello' clearly with good breath support", "type": "word", "difficulty": 2},
        {"target": "Good morning", "instructions": "Greet naturally with clear pronunciation", "type": "sentence", "difficulty": 3},
        {"target": "The sun is bright", "instructions": "Read this sentence with clear 'S' and 'T' sounds", "type": "sentence", "difficulty": 3}
    ]
    
    exercise_index = (current - 1) % len(exercises)
    exercise = exercises[exercise_index]
    
    return {
        "exercise_number": current,
        "total_exercises": 4,
        **exercise
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
    print("   Feedback Page:  http://localhost:8000/feedback")
    print("   Help Page:      http://localhost:8000/help")
    print("   API Health:     http://localhost:8000/api/health")
    print("   Static Files:   http://localhost:8000/static/")
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