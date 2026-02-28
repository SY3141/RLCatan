#!/bin/bash
# Build script for Render deployment
# This script is executed during the Render build process

echo "Starting build process..."

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing catanatron with web dependencies..."
pip install -e .[web]

echo "Build completed successfully!"

