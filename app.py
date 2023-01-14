import os
import numpy as np
from flask import Flask, request, jsonify, redirect, render_template
from flask_cors import CORS
from keras.models import load_model
from PIL import Image, ImageOps
import cv2
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
CORS(app)
app.config["imgdir"] = "Images/"

@app.route('/')
def base_main():
    address_fe = 'http://localhost:3000/' # current local front-end side
    route_status = ''

    try:
        # check whether main webapp (front-end) service is available
        response_check = requests.get(address_fe)
    except:
        route_status = 'Root url is accessed, cannot redirect the user to the front-end side. Main web service is offline'
        print(route_status)
        reroute = render_template('error_500.html')
    else:
        if response_check.status_code == 200: # redirect the user to the main website if the service is running
            route_status = 'Root url is accessed, redirect the user to the front-end side. Main web service is online'
            print(route_status)
        # redirect the user to front-end side if the base url is accessed
        reroute = redirect(address_fe)

    return reroute


# function to do classification with EfficientNetB-7 Model
@app.route('/prediction', methods=['POST'])
def predictions():
    # Load trained and tested EfficientNetB-7 model
    model_path = 'Model/EfficientNetB7_SGD.h5'
    model = load_model(model_path, compile=False)

    # Load image that sent from the front-end
    img_uploaded = request.files
    img_file = img_uploaded.get('file')
    filename_orig = secure_filename(img_file.filename) # make it easier to see what is the original file
    filename = 'identification_img.png' # save file 
    filepath = os.path.join(app.config['imgdir'], filename);
    img_file.save(filepath)
    img_loaded = Image.open(os.path.join(app.config['imgdir'], filename))
    img_loaded.save('Images/identification_img.png')

    img_dim = (150, 150)
    img_pred = cv2.imread('Images/identification_img.png') # read input image
    image_shape = cv2.resize(img_pred, img_dim) # set width and height dimension for the image
    image_pred = np.expand_dims(image_shape, axis=0) # expand the image dimension to 4 dimension
    pred_labels = ['Bukan Fundus Mata', 'Cataract', 'Normal' ]

    try:
        # do prediction to the pre-processed image
        model_pred = model.predict(image_pred)
        # assign prediction output into variable
        model_pred_output = pred_labels[np.argmax(model_pred)]
    except Exception as error:
        # print('Prediction process cannot be complete!')
        print('error: {err}'.format(err=error))
    else:
        output_str = 'Hasil Prediksi Kelas/Kategori Gambar ini adalah ' + model_pred_output
        print(output_str)

    # return this when output_str is empty when classification process failed
    if model_pred_output == 'Bukan Fundus Mata':
        output_str = "Klasifikasi tidak dapat dilakukan. Gambar tidak dikenali sebagai fundus mata. Mohon pilih file yang sesuai."
        response_string = "Image is not categorised as fundus mata, please check your file"
        file_status = "Cannot be processed"
        success_status = False
        details = {
            'processed_file': filename_orig,
            'response_string': response_string, 
            'file_Status': file_status, 
            'success_status': success_status,
            'result': output_str
            }
    # return this when the process if finished successfully
    else:
        response_string = "Process is complete"
        file_status = "Received"
        success_status = True
        details = {
            'processed_file': filename_orig,
            'response_string': response_string, 
            'file_Status': file_status, 
            'success_status': success_status,
            'result': output_str
            }
        
    return jsonify(details)