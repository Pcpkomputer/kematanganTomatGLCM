from flask import Flask, render_template, url_for, redirect, request, session
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

app.secret_key="GLCMTomat"

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="glcmkematangantomat"
)


@app.route("/logout")
def logout():
    session.pop("login")
    return redirect(url_for("login"))

@app.route("/login", methods=["POST","GET"])
def login():
    if 'login' in session:
        return redirect(url_for("index"))
    if request.method=="POST":
        email = request.form["email"]
        password = request.form["password"]

        mydb.connect()
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM user WHERE email=%s",(email,))
        row = cursor.fetchone()
        mydb.close()
        cursor.close()

        if row==None:
            return render_template("login.html",error="Akun tidak ditemukan...")
        e = row[1]
        p = row[2]

        if e==email and p==password:
            session["login"] = True
            return redirect(url_for("index"))
        else:
            return render_template("login.html",error="Login gagal...")
    return render_template("login.html")


@app.route("/logs/hapus/<id>")
def hapuslogs(id):
    if 'login' not in session:
        return redirect(url_for("login"))
    mydb.connect()
    cursor = mydb.cursor()
    cursor.execute("DELETE FROM rekamjejak WHERE id=%s",(id,))
    mydb.commit()
    cursor.close()
    mydb.close()
    
    return redirect(url_for("logs"))

@app.route("/logs/<id>")
def detaillogs(id):
    if 'login' not in session:
        return redirect(url_for("login"))
    mydb.connect()
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM rekamjejak WHERE id=%s",(id,))
    row = cursor.fetchone()
    cursor.close()
    mydb.close()

    return render_template("detaillog.html",detectpage=True,percentage=row[3],request_time=row[4],label=row[2], modelexist=True,imagelabelled = row[6], image=row[1])

@app.route("/")
def index():
    if 'login' not in session:
        return redirect(url_for("login"))
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
    if 'login' not in session:
        return redirect(url_for("login"))
    if request.method=="POST" and request.files["file"]:
        start = time.time()

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

        font = cv2.FONT_HERSHEY_SIMPLEX
     
        
        result = tensorflow.nn.softmax(logits)
        indexmax = np.argmax(result)


        if indexmax==0:
            label = "Setengah matang"
        elif indexmax==1:
            label = "Matang"
        elif indexmax==2:
            label = "Mentah"
      
        cv2.putText(image,label,(10,45), font, 1,(0,255,0),2)
        labelledid = str(uuid.uuid1())
        cv2.imwrite("static/labelled/"+labelledid+".jpg",image)

        request_time = time.time() - start
        percentage = str(float(result[0][indexmax])*100)

        mydb.connect()
        cursor=mydb.cursor()
        cursor.execute("INSERT INTO rekamjejak VALUES (NULL,%s,%s,%s,%s,%s,%s,NOW())",(id+".jpg",label,percentage,str(request_time),"Classified",labelledid+".jpg"))
        mydb.commit()
        cursor.close()


        return render_template("detecttomato.html",detectpage=True,percentage=percentage,request_time=request_time,label=label, modelexist=True,imagelabelled = labelledid+".jpg", image=id+".jpg")

    modelexist = os.path.exists(os.path.join(os.getcwd(),"model.h5"))
    return render_template("detecttomato.html",modelexist=modelexist,detectpage=False)

@app.route("/logs")
def logs():
    if 'login' not in session:
        return redirect(url_for("login"))
    mydb.connect()
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM rekamjejak")
    rows = cursor.fetchall()
    cursor.close()
    mydb.close()
    
    return render_template("detectionlogs.html",rows=rows)

@app.route("/model", methods=["POST","GET"])
def model():
    if 'login' not in session:
        return redirect(url_for("login"))
    if request.method=="POST" and request.files["file"]:
        request.files["file"].save("model.h5")
        return render_template("selectmodel.html",success="Model berhasil diupload...")
    return render_template("selectmodel.html")
if __name__=='__main__':
    app.run(debug=True)