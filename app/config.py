import os
from dotenv import load_dotenv

# Root path of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to texts with exercises and uploaded texts
DATA_PATH = os.path.join(ROOT_DIR, 'data/')