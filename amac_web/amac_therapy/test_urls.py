import requests
import time

# Start server (if not running)
# import subprocess
# server = subprocess.Popen(['python', 'main.py'])
# time.sleep(3)

urls_to_test = [
    'http://localhost:8000/',
    'http://localhost:8000/therapy',
    'http://localhost:8000/dashboard',
    'http://localhost:8000/static/therapy.html',
    'http://localhost:8000/app',
    'http://localhost:8000/api',
    'http://localhost:8000/login'
]

print('Testing URLs on localhost:8000...')
for url in urls_to_test:
    try:
        response = requests.get(url, timeout=2)
        print(f'{url}: {response.status_code}')
        if response.status_code == 200:
            print(f'  SUCCESS! Dashboard found at: {url}')
            break
    except:
        print(f'{url}: Failed to connect')
