# Garbage Classification with Keras and Flask
![yes](https://github.com/user-attachments/assets/22d48842-3115-4ddf-9091-c22b9130713c)
![no](https://github.com/user-attachments/assets/eb76007e-0440-4dda-808a-686125b9be64)
## Project Overview
This project is designed to classify waste as either recyclable or non-recyclable. It uses a Keras neural network model for image classification and Flask as the backend framework for serving a web interface. The goal is to improve the recycling process by automatically sorting waste into various categories, such as paper, plastic, glass, and more
## Features
- Classifies waste into 12 categories: paper, plastic, metal, glass (brown, green, white), shoes, clothes, batteries, organic waste, and other trash.
- Flask-based web app for uploading images of waste for classification.
- Uses a Convolutional Neural Network (CNN) for image classification.
- REST API support for easy integration with other systems.
## Technologies Used
- Python
- Keras – to build the neural network model.
- TensorFlow – as the backend for Keras.
- Flask – to run the web application.
- PIL – for image processing.
- Matplotlib – for visualization
- HTML/CSS – for the frontend interface.
## Dataset
https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data
If you are the owner of any images and would like them removed from the project, please contact us.
## Setup
Below are the steps to run the application using Docker.
### Prerequisites
Make sure you have Docker installed on your computer.
### Step 1: Clone the repository
```bash
git clone https://github.com/72Clooud/GarbageClassification-keras-flask--for-recycling-not-for-recycling-.git
cd GarbageClassification-keras-flask--for-recycling-not-for-recycling-
```
### Step 2: Build the Docker image
`docker build -t garbage-classification .`
### Step 3: Run the container
`docker run -p 5000:5000 garbage-classification`