import pyttsx3
from flask import Flask, request, jsonify
import torch
import numpy as np
import cv2, base64, os, sys

from src.model import ASLMLP
from src.extract_landmarks import extract_landmark

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ASLMLP()
model.load_state_dict(torch.load("../model/asl_landmarks_model.pth", map_location=device))
model.eval().to(device)

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

nothing_count = 0
stack = []
@app.route('/predict', methods=['POST'])
def predict():
    try:
        global nothing_count
        data = request.json['image']
        img_bytes = base64.b64decode(data.split(',')[0])
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        landmarks = extract_landmark(img)

        if landmarks is None:
            nothing_count += 1
            print('NC:', nothing_count)
            if nothing_count >= 3:
                print(stack)
                engine.say(stack.__str__())
                engine.runAndWait()
                stack.clear()

                return jsonify({
                    "prediction": 'Nothing',
                    "confidence": 1
                })
        else:
            x = torch.tensor(landmarks, dtype=torch.float32).to(device).unsqueeze(0)
            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)
                conf, idx = torch.max(probs, 1)
                label = labels[idx.item()]
            print(label)

            nothing_count = 0
            if label == 'del':
                stack.pop()
            elif label == 'space':
                stack.append(" ")
            else:
                stack.append(label)

        return jsonify({
            "prediction": label,
            "confidence": round(conf.item(), 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/')
def home():
    return "ASL backend is running!"


if __name__ == "__main__":
    app.run(debug=True)
