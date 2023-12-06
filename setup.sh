echo "Activating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
echo "Virtual environment activated, installing dependencies..."
pip install --upgrade pip
pip install -r requirements.conf --progress-bar off
echo "Dependencies installed."