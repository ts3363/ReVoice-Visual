from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

os.chdir('.')  # Serve from current directory

print("Starting AMAC Therapy server at http://localhost:8080")
print("Access:")
print("  Login: http://localhost:8080/login.html")
print("  Dashboard: http://localhost:8080/dashboard.html")

server = HTTPServer(('localhost', 8080), SimpleHTTPRequestHandler)
server.serve_forever()
