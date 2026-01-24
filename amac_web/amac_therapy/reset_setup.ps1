Write-Host "Setting up AMAC Therapy System..." -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Check current directory
$currentDir = Get-Location
Write-Host "Current directory: $currentDir" -ForegroundColor Cyan

# Check for required files
$requiredFiles = @("main.py", "requirements.txt", "app\api\therapy_endpoints.py", "app\static\therapy.html")
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ Found: $file" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing: $file" -ForegroundColor Red
    }
}

# Check virtual environment
if (Test-Path "venv") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
} else {
    Write-Host "⚠️  Virtual environment not found" -ForegroundColor Yellow
    Write-Host "   Run: python -m venv venv" -ForegroundColor White
}

Write-Host "`n📋 Quick Start Commands:" -ForegroundColor Magenta
Write-Host "   1. Activate venv: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   2. Start server: python main.py" -ForegroundColor White
Write-Host "   3. Test API: curl http://localhost:8000/api/health" -ForegroundColor White
Write-Host "   4. Open therapy: start http://localhost:8000/static/therapy.html" -ForegroundColor White

Write-Host "`n🔧 Troubleshooting:" -ForegroundColor Yellow
Write-Host "   If port 8000 is in use: netstat -ano | findstr :8000" -ForegroundColor White
Write-Host "   Kill process: taskkill /PID <PID> /F" -ForegroundColor White
