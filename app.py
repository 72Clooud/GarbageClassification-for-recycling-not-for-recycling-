from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import PIL
import pickle
import os
import keras

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')

#model = pickle.load(open('models/model7008.pkl', 'rb'))
model = keras.models.load_model('./models/model.keras')


# Limiting the size of uploading files 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
# Static Folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
EMOTION = {
    0: "surprised",
    1: "scared",
    2: "disgust",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "neutral"
}


def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def root():
    return render_template('index.html')

@app.route("/upload_image", methods=['POST'])
def upload_image():
    file = request.files['file']
    if file.filename == "":
        return redirect(request.url)
    if file and allowed_file(file.filename):
        file_name = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(file_path)
        return redirect(url_for('display_and_predict', file_name=file_name))
     
    return "error"
    
    
@app.route("/display_and_predict", methods=['GET', 'POST'])
def display_and_predict():
    
    file_name = request.args.get('file_name')
    prediction = None
    
    if request.method == 'POST':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        
        image = PIL.Image.open(file_path)
        image = image.resize((100, 100))
        image = np.array(image).reshape(1, 100, 100, 3)
        
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        
        prediction = f'Prediction: {EMOTION[predicted_class[0]]}'
    
    return render_template('display_and_predict.html', prediction=prediction, file_name=file_name)


if __name__ == ('__main__'):
    app.run(host='0.0.0.0', debug=True)