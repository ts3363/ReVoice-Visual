// logout.js - Add logout functionality to AMAC Therapy dashboard
(function() {
    console.log('AMAC Therapy: Loading logout functionality...');
    
    // Wait for page to load
    setTimeout(function() {
        // Create logout button
        const logoutBtn = document.createElement('button');
        logoutBtn.id = 'amacLogoutBtn';
        logoutBtn.innerHTML = 'Logout';
        logoutBtn.style.cssText = \
            position: fixed;
            top: 20px;
            right: 20px;
            background: #2c5aa0;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            z-index: 1000;
            font-weight: 600;
            font-family: 'Segoe UI', system-ui, sans-serif;
            font-size: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            transition: all 0.3s;
        \;
        
        // Hover effect
        logoutBtn.onmouseover = function() {
            this.style.backgroundColor = '#1e4280';
            this.style.transform = 'translateY(-2px)';
        };
        logoutBtn.onmouseout = function() {
            this.style.backgroundColor = '#2c5aa0';
            this.style.transform = 'translateY(0)';
        };
        
        // Logout function
        logoutBtn.onclick = function() {
            if (confirm('Are you sure you want to logout from AMAC Therapy?')) {
                // Clear all login data
                localStorage.removeItem('amacLoggedIn');
                localStorage.removeItem('amacUsername');
                sessionStorage.removeItem('amacLoggedIn');
                sessionStorage.removeItem('amacUsername');
                
                // Redirect to login page
                window.location.href = 'login.html';
            }
        };
        
        // Add button to page
        document.body.appendChild(logoutBtn);
        console.log('AMAC Therapy: Logout button added to dashboard');
        
        // Check if user is logged in
        const isLoggedIn = localStorage.getItem('amacLoggedIn') || sessionStorage.getItem('amacLoggedIn');
        if (!isLoggedIn && !window.location.href.includes('login.html')) {
            console.log('AMAC Therapy: User not logged in, redirecting to login...');
            // Show message before redirecting
            alert('Please login to access the AMAC Therapy dashboard');
            window.location.href = 'login.html';
        } else {
            // Update button text with username if available
            const username = localStorage.getItem('amacUsername') || sessionStorage.getItem('amacUsername') || 'User';
            logoutBtn.innerHTML = \Logout (\)\;
        }
        
    }, 1000); // Wait 1 second for page to load
    
})();
