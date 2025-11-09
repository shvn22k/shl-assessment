import os
import sys

# Add project root and src to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'src'))

# Import the FastAPI app
from app import app

# Vercel uses ASGI for FastAPI
from mangum import Mangum

handler = Mangum(app)

