from flask import Flask, request, render_template, redirect, url_for
import pickle
import os

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')

#model = pickle.load(open('models/model6515.pkl'), 'rb')

app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:        
        file_name = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(file_path)
        return redirect(url_for('display_image', file_name=file_name))
    
    return "error"
    
    
@app.route('/display_image')
def display_image():
    file_name = request.args.get('file_name')
    return render_template('display_image.html', file_name=file_name)


@app.route('predict')
def predict():
    


if __name__ == ('__main__'):
    app.run(host='0.0.0.0', debug=True)