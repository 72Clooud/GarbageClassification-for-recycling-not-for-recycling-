import numpy as np
import os

import torch
from torch import nn 
from torchvision import transforms

from model_handler import ModelHandler
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    
    def __init__(self):
        self.allowed_extensions = ['txt', 'pdf', 'png', 'jpg', 'jpeg']
        self.categories = { 0: "battery", 1: "biological", 2: "brown-glass", 3: "cardboard", 4: "clothes",
                           5: "green-glass", 6: "metal", 7: "paper", 8: "plastic", 9: "shoes", 10: "trash", 11: "white-glass"}
        self.recycling = [0, 2, 3, 4, 5, 6, 7, 8, 11]
        self.model_path = '../models/best_model.pth'
        self.num_calsses = len(self.categories)

class AppModelHandler:
    
    def __init__(self, model_path: str, device):
        self.model_path = model_path
        self.device = device
        self.model_handler = ModelHandler(config.num_calsses, device)
        self.model_handler.load_model(model_path)
        self.model = self.model_handler.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def predict(self, input_image: torch.Tensor) -> int:
        with torch.no_grad():
            prediction = self.model(input_image)
            softmax = nn.Softmax(dim=1)
            probabilities  = softmax(prediction)
            predicted_class = torch.argmax(probabilities , dim=1).item()
        return predicted_class
            
    def process_input_data(self, input_data_path):
        img = Image.open(input_data_path).convert('RGB')
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img.to(self.device)
        
        
config = Config()

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = 'static/uploads'

def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in config.allowed_extensions

@app.route("/")
def root():
    return render_template('index.html')

@app.route("/upload_image", methods=['POST'])
def upload_image():
    file = request.files['file']
    file_name = file.filename
    if file.filename == "":
        return redirect(request.url)
    if file and allowed_file(file_name):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(file_path)
        return redirect(url_for('display_and_predict', file_name=file_name))
    return "error"
    
    
@app.route("/display_and_predict", methods=['GET', 'POST'])
def display_and_predict():
    
    file_name = request.args.get('file_name')
    recycling_message = None
    prediction = None
    predicted_class = None
    
    if request.method == 'POST':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        
        model_handler = AppModelHandler(config.model_path, device)
        img = model_handler.process_input_data(file_path)
        prediction = model_handler.predict(img)
        predicted_class = config.categories[prediction]
        
        if prediction in config.recycling:
            recycling_message = 'Suitable for recycling'
        else:
            recycling_message = 'Not suitable for recycling'
        
    return render_template('display_and_predict.html', predicted_class=predicted_class, recycling_message=recycling_message, file_name=file_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)