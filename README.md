# Waste Classifier AI ♻️

A Flask web application that classifies waste images into categories (cardboard, glass, metal, paper, plastic, trash) using a fine-tuned ResNet50 model.

## Features
- User registration, login, and profile
- Upload image → instant classification result
- Prediction history
- Responsive UI

## Screenshots
![Registration](static/IMAGES/REGISTRATION%20PAGE.jpg)
![Login](static/IMAGES/LOGIN%20PAGE.jpg)
![Upload](static/IMAGES/UPLOAD%20PAGE.jpg)
![Classification History](static/IMAGES/CLASSIFICATION_HISTORY%20PAGE.jpg)
![Profile](static/IMAGES/PROFILE%20PAGE.jpg)

## How to Run
```bash
pip install flask torch torchvision pillow
python app.py