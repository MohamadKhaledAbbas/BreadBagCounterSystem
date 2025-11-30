from fastapi.templating import Jinja2Templates
from src.logging.Database import DatabaseManager

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This file's directory
templates = Jinja2Templates("src/endpoint/templates")

db = DatabaseManager("data/db/bag_events.db")