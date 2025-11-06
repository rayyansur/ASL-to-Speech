import cv2
import torch
import mediapipe as mp
import os
from model import ASLMLP
from extract_landmarks import extract_landmark

mp_hands = mp.solutions.hands

DATASET_DIR = "../data/asl_alphabet_split/test/"
MODEL_DIR = "../model/asl_landmarks_model.pth"
model = ASLMLP()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_DIR, map_location=device))

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
            if img is not None:
                norm_landmarks = extract_landmark(img)
                if norm_landmarks is None:
                    if label == "nothing":
                        passCount += 1
                    else:
                        noCount += 1
                else:
                    input_tensor = torch.tensor(norm_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
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