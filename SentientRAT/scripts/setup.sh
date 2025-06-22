
#!/bin/bash

# Create directories
mkdir -p knowledge memory logs gui/src models

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip docker.io nodejs npm

# Install Python dependencies
pip install -r requirements.txt

# Install React dependencies and build GUI
cd gui
npm install
npm run build
cd ..

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

echo "Setup complete. Run 'python main_agent.py' to start SentientRAT."