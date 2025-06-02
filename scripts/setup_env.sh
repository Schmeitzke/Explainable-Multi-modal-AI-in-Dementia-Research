#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$( dirname "$SCRIPT_DIR" )"

cd "$ROOT_DIR"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux operating system"
    PYTHON_CMD="python3"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS operating system"
    PYTHON_CMD="python3"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows operating system"
    PYTHON_CMD="python"
else
    echo "Unknown operating system: $OSTYPE"
    echo "Defaulting to python3 command"
    PYTHON_CMD="python3"
fi

if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: $PYTHON_CMD could not be found. Please install Python first."
    exit 1
fi

if ! $PYTHON_CMD -m pip show virtualenv &> /dev/null; then
    echo "Installing virtualenv..."
    $PYTHON_CMD -m pip install virtualenv
fi

echo "Creating virtual environment 'env' in the root directory..."
$PYTHON_CMD -m virtualenv env

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source env/Scripts/activate
else
    source env/bin/activate
fi

echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! To activate the environment, run:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "source env/Scripts/activate"
else
    echo "source env/bin/activate"
fi