import os
import re

# Path to the new frontend file
html_path = r"amac_web/amac_therapy/app/static/therapy.html"

print(f"[*] Scanning {html_path} for connection logic...")

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Search for WebSocket connections
    ws_matches = re.findall(r'new WebSocket\([\'"`](.*?)[\'"`]\)', content)
    print("\n--- WEBSOCKET TARGETS FOUND ---")
    if ws_matches:
        for match in ws_matches:
            print(f"  Found: {match}")
    else:
        print("  [!] No direct 'new WebSocket' calls found.")

    # 2. Search for fetch/API calls
    fetch_matches = re.findall(r'fetch\([\'"`](.*?)[\'"`]', content)
    print("\n--- API ENDPOINTS CALLED ---")
    for match in fetch_matches:
        print(f"  Found: {match}")
        
    # 3. Search for specific port numbers (maybe she hardcoded port 5000?)
    port_matches = re.findall(r':\d{4}', content)
    print("\n--- HARDCODED PORTS ---")
    for match in list(set(port_matches)): # unique only
        print(f"  Found: {match}")

else:
    print(f"[!] Could not find file at {html_path}")