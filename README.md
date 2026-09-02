# Food Freshness & Quality Classifier

A computer vision web application that classifies food and produce quality (fresh, okay, or avoid) from user-uploaded images using pre-trained Hugging Face vision models.

## Key Features

- Single-image quality classification (`Fresh`, `Okay`, `Avoid`)
- Confidence score reporting
- Lightweight REST API for inference
- Clean web interface for drag-and-drop image uploads

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Machine Learning**: Hugging Face Transformers, PyTorch, Pillow
- **Frontend**: HTML5, CSS3, JavaScript

## Project Structure

```text
food-freshness-classifier/
├── app/
│   ├── main.py
│   └── model.py
├── static/
│   └── index.html
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
