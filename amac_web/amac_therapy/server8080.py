import http.server
import socketserver
import os

PORT = 8080
os.chdir('.')

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/login.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

print(f"Starting AMAC Therapy server on port {PORT}")
print("Access: http://localhost:8080")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
