# -*- coding: utf-8 -*-
"""
FastAPI Persistent Inference Service - Main Entry Point
This script is the main entry point for the FastAPI service.
It imports the application from inference_app.py and runs it.
"""
import os
import sys
import logging
from datetime import datetime

# Set up logging before importing the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set a start time environment variable for the health check
os.environ['START_TIME'] = str(datetime.now())

# Check for model paths in command line arguments and set environment variables
if len(sys.argv) > 2:
    MODEL_PATH = sys.argv[1]
    LABEL_PATH = sys.argv[2]
    os.environ['MODEL_PATH'] = MODEL_PATH
    os.environ['LABEL_PATH'] = LABEL_PATH
    logger.info(f"Using command line arguments for model paths: {MODEL_PATH}, {LABEL_PATH}")

# The actual FastAPI application logic is now in inference_app.py
# We run the application directly from uvicorn, pointing to the app instance in inference_app.py
# This structure is cleaner and allows for easier testing/refactoring of components.

# We will use a shell command to run the application to ensure it starts correctly.
# The user's original intent was to split the file, which we have done.
# The new Inference.py will just be a placeholder/entry point, and the user will run the service using:
# python Inference.py [model_path] [label_path]

# Since this file is meant to be executed, we don't need to put the uvicorn.run call here
# as the user will typically run it via the command line, e.g., 'uvicorn inference_app:app --host 0.0.0.0 --port 8000'
# However, to maintain the original file's execution capability, we will include the uvicorn run block.

if __name__ == "__main__":
    import uvicorn
    from inference_app import app
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)