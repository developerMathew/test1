from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import cv2
import numpy as np
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.eval()

@app.route('/', methods=['GET', 'POST'])
def upload():
    detections = []
    result_path = None
    if request.method == 'POST':
        conf_thresh = float(request.form.get('threshold', 0.25))

        file = request.files['file']
        if not file:
            return "No file uploaded", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        results = model(filepath)
        results = results.pandas().xyxy[0]
        results = results[results['confidence'] >= conf_thresh]

        img = Image.open(filepath)
        result_img = np.array(img)

        for _, row in results.iterrows():
            label = f"{row['name']} {row['confidence']:.2f}"
            detections.append({
                'class': row['name'],
                'confidence': round(row['confidence'], 2),
                'x': int(row['xmin']),
                'y': int(row['ymin']),
                'w': int(row['xmax'] - row['xmin']),
                'h': int(row['ymax'] - row['ymin']),
            })
            # Draw box
            cv2.rectangle(result_img, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (255, 0, 0), 2)
            cv2.putText(result_img, label, (int(row['xmin']), int(row['ymin']) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        output_filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(RESULT_FOLDER, output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        result_path = f"results/{output_filename}"

    return render_template("index.html", result_path=result_path, detections=detections)

if __name__ == '__main__':
    app.run(debug=True)
