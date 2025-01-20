# AI Meme Generator

## Introduction
This project is an AI-powered meme generator that allows users to upload images, generates witty captions using AI, and creates memes that can be downloaded.

## Architecture Overview
The application is built using Flask for the web interface, TensorFlow/Keras for image processing, and Hugging Face Transformers for caption generation.

## Setup Instructions
1. Install Python and the required libraries from `requirements.txt`.
2. Run the Flask application using `python app.py`.
3. Access the application in your web browser at `http://localhost:5000`.

## Codebase Structure
- `app.py`: Main Flask application file.
- `requirements.txt`: List of project dependencies.
- `static/`: Directory containing static files like CSS and JavaScript.
- `templates/`: Directory containing HTML templates.

## Testing Instructions
Upload an image through the web interface and verify that the application generates captions and creates a meme.
