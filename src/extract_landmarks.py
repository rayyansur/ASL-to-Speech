import os
import csv
import cv2
import mediapipe as mp
import time
from normalize_landmarks import normalize

mp_hands = mp.solutions.hands

DATASET_DIR = "../data/asl_alphabet_split/train/"
OUTPUT_CSV = "normalized_landmarks.csv"

def extract_landmarks():
    with open(OUTPUT_CSV, mode='w', newline='') as f:

        writer = csv.writer(f)
        header = [f'{axis}{i}' for i in range(21) for axis in ['x','y','z']] + ['label']
        writer.writerow(header)
        start_time = time.perf_counter()

        count = 0
        total = 0

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
            for label in os.listdir(DATASET_DIR):
                print(label)
                label_path = os.path.join(DATASET_DIR, label)
                if not os.path.isdir(label_path):
                    continue
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    rgb = cv2.convertScaleAbs(rgb, alpha=1.2, beta=20)
                    results = hands.process(rgb)
                    if results.multi_hand_landmarks:
                        hand = results.multi_hand_landmarks[0]
                        coords = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                        norm_coords = normalize(coords)
                        row = [val for tup in norm_coords  for val in tup]
                        row.append(label)
                        writer.writerow(row)
                        count +=1

                    total += 1

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

    print("Landmark data saved to", OUTPUT_CSV, f" in {elapsed_time:.2f}  seconds.")
    print(f"Accuracy  ${count / total * 100}%")


if __name__ == "__main__":
    extract_landmarks()

