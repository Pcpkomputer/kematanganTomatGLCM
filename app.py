from flask import Flask, render_template, url_for, redirect, request
import os
import mysql.connector
import psutil
import time
from utils.feature_extraction import calc_glcm_all_agls
import cv2
import numpy as np
import uuid
from PIL import Image
import tensorflow
from tensorflow import keras

app = Flask(__name__)


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="glcmkematangantomat"
)

@app.route("/")
def index():
    start = time.time()

    mydb.connect()
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM rekamjejak")
    rows = cursor.fetchall()
    cursor.close()
    mydb.close()
 
    request_time = time.time() - start

    return render_template("index.html",rows=rows, cpuusage=psutil.cpu_percent(),memoryusage=psutil.virtual_memory()[2],requesttime=request_time)

@app.route("/detection", methods=["POST","GET"])
def detection():
    if request.method=="POST" and request.files["file"]:
        image = request.files["file"]
        id = str(uuid.uuid1())
        image.save("static/classification/"+id+".jpg")
        # PROSES PREPROCESSING
        image = cv2.imread("static/classification/"+id+".jpg")
        #image = image[1408:1408+2208,0:0+2176]
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (300,500))

        properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
        feature = calc_glcm_all_agls(resize, "undefined", props=properties)

        model = keras.models.load_model("model.h5")
        logits = model.predict([feature])
        
        print(tensorflow.nn.softmax(logits))

        return "123"

    modelexist = os.path.exists(os.path.join(os.getcwd(),"model.h5"))
    return render_template("detecttomato.html",modelexist=modelexist)

@app.route("/logs")
def logs():
    mydb.connect()
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM rekamjejak")
    rows = cursor.fetchall()
    cursor.close()
    mydb.close()

    return render_template("detectionlogs.html",rows=rows)

@app.route("/model", methods=["POST","GET"])
def model():
    if request.method=="POST" and request.files["file"]:
        request.files["file"].save("model.h5")
        return render_template("selectmodel.html",success="Model berhasil diupload...")
    return render_template("selectmodel.html")
if __name__=='__main__':
    app.run(debug=True)