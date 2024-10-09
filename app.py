from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('index.html')

if __name__ == ('__main__'):
    app.run(host='0.0.0.0', debug=True)