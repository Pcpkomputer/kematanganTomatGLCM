from flask import Flask, render_template, url_for, redirect

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detection")
def detection():
    return render_template("detecttomato.html")

@app.route("/logs")
def logs():
    return render_template("detectionlogs.html")

if __name__=='__main__':
    app.run(debug=True)