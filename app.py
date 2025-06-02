import argparse
import io
import os
import base64
import time
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])  # Trust the custom model class

# Initialize YOLO model
model = YOLO('best.pt')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
PORT_NUMBER = 5000

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            key = cv2.waitKey(50) & 0xFF
            if key == 27 or key == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('first'))

@app.route('/')
@app.route('/first')
def first():
    return render_template("first.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/image')
def image():
    return render_template("image.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('image'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('image'))

    upl_img = Image.open(file)
    result = model.predict(source=upl_img)[0]
    res_img = Image.fromarray(result.plot())
    image_byte_stream = io.BytesIO()
    res_img.save(image_byte_stream, format='PNG')
    image_byte_stream.seek(0)
    image_base64 = base64.b64encode(image_byte_stream.read()).decode('utf-8')

    return render_template('image.html', detection_results=image_base64)

@app.route("/video")
def video():
    return render_template('video.html')

@app.route("/predict_img", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST" and 'file' in request.files:
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        file_extension = f.filename.rsplit('.', 1)[1].lower()

        if file_extension == 'mp4':
            cap = cv2.VideoCapture(filepath)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 50.0, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, save=True)
                res_plotted = results[0].plot()
                out.write(res_plotted)
                cv2.imshow("result", res_plotted)

                if (cv2.waitKey(25) & 0xFF) == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            return redirect(url_for('video'))

    return render_template('video.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop():
    return render_template('first.html')

@app.route('/performance')
def performance():
    return render_template("performance.html")

@app.route('/chart')
def chart():
    return render_template("chart.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
