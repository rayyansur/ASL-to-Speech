import cv2
import time
import requests
import base64

# Your Flask endpoint (runs on same machine usually)
API_URL = "http://127.0.0.1:5000/predict"

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

print("Sending frames to Flask every 1s — press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Encode image as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Send to Flask
    try:
        response = requests.post(API_URL, json={"image": jpg_as_text})
        if response.ok:
            print(response.json().get("prediction"))
        else:
            print("Error:", response.status_code)
    except Exception as e:
        print("Failed to send frame:", e)

    # Wait 1 second before next shot
    time.sleep(1)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
