Write-Host "Debugging AMAC API..." -ForegroundColor Cyan

# Test API endpoints
$endpoints = @(
    "http://localhost:8000/",
    "http://localhost:8000/api/health",
    "http://localhost:8000/api/user/profile",
    "http://localhost:8000/static/therapy.html"
)

foreach ($endpoint in $endpoints) {
    Write-Host "`nTesting: $endpoint" -ForegroundColor Yellow
    try {
        $response = Invoke-WebRequest -Uri $endpoint -ErrorAction Stop
        Write-Host "   ✅ Status: $($response.StatusCode)" -ForegroundColor Green
        if ($endpoint -like "*api*") {
            $json = $response.Content | ConvertFrom-Json
            Write-Host "   Response: $($json | ConvertTo-Json -Compress)" -ForegroundColor Gray
        }
    } catch {
        Write-Host "   ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}
