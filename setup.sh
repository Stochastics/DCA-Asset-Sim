#!/bin/bash

# Script to set up the environment for the DCA simulation program

echo "Setting up the environment for DCA simulation..."

# Update package manager and install Python3 and pip
echo "Updating package manager and installing Python3 and pip..."
sudo apt update && sudo apt install -y python3 python3-pip
echo "Upgrading pip to the latest version..."
pip install --upgrade pip --timeout 300
echo "Creating a virtual environment..."
python3 -m venv dca_env
source dca_env/bin/activate

echo "Installing required Python packages from requirements.txt..."
pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ --timeout 300

echo "Setup complete. To start using the environment, run:"
echo "source dca_env/bin/activate"
