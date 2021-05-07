from flask import Flask, render_template, url_for, redirect, request
import os
import mysql.connector
import psutil
import time

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

@app.route("/detection")
def detection():
    modelexist = os.path.exists(os.path.join(os.getcwd(),"model.joblib"))
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
        request.files["file"].save("model.joblib")
        return "damn"
    return render_template("selectmodel.html")
if __name__=='__main__':
    app.run(debug=True)