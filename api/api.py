import os
import time
from flask import Flask, flash, request, redirect, url_for, session, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
import numpy
import matplotlib.pyplot as plt
from os.path import join, dirname, realpath
import subprocess
from compare import localize_image 

fileList = ["orb_extractor.cc", "orb_params.cc", "orb_extractor_node.cc", "main.cc"]

BASE_PATH = dirname(realpath(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])



app = Flask(__name__)
app.config['BASE_PATH'] = BASE_PATH


@app.route('/')
def get_current_time():

    return {'time': time.time()}

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['BASE_PATH'], "images/img.jpg"))
    print(file)
    
    cmd = "cpy.cc"

    pkg_config_exe = os.environ.get('PKG_CONFIG', None) or 'pkg-config'

    subprocess.call("clang++ -std=c++11 orb_extractor.cc orb_params.cc orb_extractor_node.cc main.cc `pkg-config --libs --cflags opencv` -lm", shell=True)
    p = subprocess.Popen("./a.out",stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode("utf-8")    

    with open("out.txt", "w") as text_file:
        text_file.writelines([str(line) for line in out])

    localize_image()

    #print(out)


    return filename

@app.route('/img/<name>')
def get_image(name):
    filename = os.path.join(app.config['BASE_PATH'], "images/img.jpg")
    return send_file(filename, mimetype='image/png')

@app.route('/plot/<name>')
def get_plot(name):
    filename = os.path.join(app.config['BASE_PATH'],'plots/plot.png')
    return send_file(filename, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",use_reloader=False)

CORS(app, expose_headers='Authorization')





