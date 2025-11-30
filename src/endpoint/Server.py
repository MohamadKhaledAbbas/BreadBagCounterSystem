from fastapi import FastAPI
from src.endpoint.routes import Analytics
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()
app.include_router(Analytics.router)

os.makedirs("data/unknown/", exist_ok=True)  # Creates the directory if it doesn't exist
app.mount("/unknown_classes", StaticFiles(directory="data/unknown/"), name="unknown_classes")

os.makedirs("data/classes/", exist_ok=True)  # Creates the directory if it doesn't exist
app.mount("/known_classes", StaticFiles(directory="data/classes/"), name="known_classes")
