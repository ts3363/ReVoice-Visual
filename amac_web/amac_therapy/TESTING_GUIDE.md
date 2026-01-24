# Testing Guide - User-Specific Dashboard

## ✅ How to Check if the Updates Are Working

### Step 1: Start the Server

Open a terminal/PowerShell in the `amac_web/amac_therapy` directory and run:

```bash
python main.py
```

**OR** if you're using the virtual environment:

```bash
# On Windows
venv\Scripts\activate
python main.py

# On Linux/Mac
source venv/bin/activate
python main.py
```

You should see:
```
🚀🚀🚀... AMAC THERAPY SERVER STARTING 🚀🚀🚀
📌 ACCESS LINKS:
   Login Page:     http://localhost:8000/login
   ...
```

### Step 2: Open the Login Page

Open your browser and go to:
- **http://localhost:8000/login** (if main.py serves it)
- **OR** directly open `login.html` in your browser (if accessing files directly)

### Step 3: Test Login with Different Usernames

#### Test Case 1: Login with "a"
1. Enter username: `a`
2. Enter any password: `123` (or anything)
3. Click "Sign In to Dashboard"
4. ✅ **Expected**: Should login successfully and redirect to dashboard

#### Test Case 2: Login with "krish"
1. Enter username: `krish`
2. Enter any password: `abc` (or anything)
3. Click "Sign In to Dashboard"
4. ✅ **Expected**: Should login successfully

#### Test Case 3: Login with any username
1. Try any username like `test123`, `john`, etc.
2. Enter any password
3. ✅ **Expected**: Should accept any username

### Step 4: Check the Dashboard

After logging in, you should see the dashboard. **Verify these things:**

#### ✅ Username Display
- Look at the top section (User Info)
- **Expected**: Should show the username you logged in with (e.g., "a", "krish", etc.)
- **NOT**: Should NOT show "Test User"

#### ✅ Profile Section
- Scroll to "Patient Profile" section
- **Expected**: 
  - Name should match your logged-in username
  - Member Since should show today's date or recent date
  - Current Level might show "Beginner" for new users

#### ✅ Statistics (for new users)
For a first-time user (like "a"), you should see:
- **Total Sessions**: 0
- **Best Score**: 0
- **Day Streak**: 0
- **Avg Score**: 0

This is **CORRECT** for a new user!

#### ✅ Recent Activity
- Should show: "No sessions yet. Start your first therapy session!"

### Step 5: Test Multiple Users (Data Isolation)

1. **Login as "a"**
   - Check the dashboard
   - Note the data (all zeros for new user)
   - Logout

2. **Login as "krish"**
   - Check the dashboard
   - Should see "krish" in the username
   - This is a separate user with separate data

3. **Login as "a" again**
   - Should show "a"'s data again
   - Each user's data is isolated

### Step 6: Verify API is Working

Open browser Developer Tools (F12) and check the Console tab:

1. You should see: `User logged in as: [your-username]`
2. Check Network tab - should see API calls to:
   - `http://localhost:8000/api/user/profile?username=...`
   - `http://localhost:8000/api/user/progress?username=...`
   - `http://localhost:8000/api/therapy/sessions?user_id=...`

### Troubleshooting

#### ❌ Dashboard shows "Test User"
- **Problem**: Dashboard might be cached or not loading properly
- **Solution**: 
  - Hard refresh (Ctrl+F5 or Ctrl+Shift+R)
  - Clear browser cache
  - Check browser console for errors

#### ❌ API calls failing (CORS errors)
- **Problem**: Server might not be running
- **Solution**: Make sure `main.py` is running on port 8000

#### ❌ "Please login" message on dashboard
- **Problem**: Login state not saved
- **Solution**: Make sure you clicked "Sign In to Dashboard" after entering credentials

#### ❌ Port 8000 already in use
- **Solution**: The server will automatically try port 8080
- Update API_BASE_URL in dashboard.html if needed, OR
- Kill the process using port 8000

### Quick Test Checklist

- [ ] Server starts without errors
- [ ] Login page opens
- [ ] Can login with username "a"
- [ ] Can login with username "krish"
- [ ] Can login with any username
- [ ] Dashboard shows correct username (not "Test User")
- [ ] Dashboard shows user-specific profile
- [ ] New users show 0 sessions/stats (correct behavior)
- [ ] Each username has separate data
- [ ] API calls are successful (check browser console)

---

## 🎯 Expected Behavior Summary

**Before (Old Behavior):**
- Dashboard always showed "Test User"
- All users saw the same hardcoded data
- Stats: Total Sessions: 5, Best Score: 85, etc. (always same)

**After (New Behavior):**
- Dashboard shows the actual logged-in username
- Each user has their own data
- New users start with 0 sessions/stats
- Data updates as users complete sessions

---

**If everything above works correctly, the update is successful! ✅**
