import random

import cv2
import torch
import mediapipe as mp
import csv
import os
from model import LandmarkModel
from normalize_landmarks import normalize
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands

DATASET_DIR = "../data/asl_alphabet_split/test/"
MODEL_DIR = "../model/asl_landmarks_model.pth"
model = LandmarkModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_DIR, map_location=device))

#labels = ['A','B','C','D', 'del', 'E','F','G','H','I','J','K','L','M','N', 'nothing', 'O','P','Q','R','S', 'space', 'T','U','V','W','X','Y','Z']
labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

passCount = 0
failCount = 0
noCount = 0

with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    for label in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label)
        print(label)
        if not os.path.isdir(label_path):
            continue
        for img_path in os.listdir(label_path):
            img = cv2.imread(os.path.join(label_path, img_path))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = cv2.convertScaleAbs(rgb, alpha=1.2, beta=20)

            results = hands.process(rgb)
            if results.multi_hand_landmarks == None:
                if label == "nothing":
                    passCount += 1
                else:
                    noCount += 1

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                coords = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                norm_coords = normalize(coords)
                input_tensor = torch.tensor(norm_coords, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model.forward(input_tensor)
                    labelOf = labels[pred.argmax().item()]


                    if labelOf == label:
                        passCount += 1
                    else:
                        failCount += 1

    print(f"Fail count: {failCount}")
    print(f"Pass count: {passCount}")
    print(f"No count: {noCount}")
    print(f"Accuracy: {(passCount/(failCount + passCount)) * 100}%")
    print(f"Total Accuracy: {(passCount/(passCount + noCount + failCount)) * 100}%")