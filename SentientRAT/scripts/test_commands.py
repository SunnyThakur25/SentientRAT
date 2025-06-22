
import requests

# Test natural language inputs
test_commands = [
    "Use nmap to scan 192.168.1.5 with aggressive mode and visualize",
    "Summarize all my pentesting books",
    "Search for recent CVEs for Apache",
    "Use sqlmap to scan example.com and show results on a chart",
    "Use hydra to brute force SSH on 192.168.1.5 with username admin",
    "Switch to DeepSeek R1 and scan 192.168.1.5 with nmap"
]

# Send requests to FastAPI endpoint
url = "http://localhost:8000/process_command"
headers = {"Authorization": "Bearer dummy_token"}

for cmd in test_commands:
    response = requests.post(url, json={"user_input": cmd}, headers=headers)
    print(f"Command: {cmd}")
    print(f"Response: {response.json()}\n")