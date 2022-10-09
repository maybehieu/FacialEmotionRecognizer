import torch
from model import myModel
import cv2
import os
from PIL import Image
import numpy as np
from dataloader import image_transform

LAST_CPT_DIR = "/checkpoints/mobilenetv2/model_weights.pth"  # path to the checkpoint weights you want to use
NUM_CLASSES = 8
width = 224
height = 224
class_dict = {
    0: 'neutral',
    1: 'happiness',
    2: 'surprise',
    3: 'sadness',
    4: 'anger',
    5: 'disgust',
    6: 'fear',
    7: 'contempt'
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# time between each time the model predict with input image
set_time = 10 if device == 'cuda' else 50
# NOTE: 'cpu' now lag-free if use Mobilenetv2 implementation (mistakes were made from my end, it was lag-free from the beginning)
model = myModel(NUM_CLASSES, "pred").to(device)
model.load_state_dict(torch.load(LAST_CPT_DIR))

print("Finished loading model.")
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# init val for classifier
label = ""
confidence = ""


# get the label and confidence through our model
def predict(image):
    model.eval()

    img = image
    input = image_transform(img)
    input = input.unsqueeze(0)
    input = input.to(device)

    with torch.no_grad():
        output = model(input)
    prob, pred_label = torch.max(output, dim=1)
    pred_label = pred_label.cpu().numpy()
    prob = prob.cpu().numpy()
    pred = class_dict[pred_label[0]]
    confidence = round(prob[0], 4)
    return pred, confidence


while(True):
    # somethign
    time += 1
    (ret, frame) = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 4)
    faces_rect = face_cascade.detectMultiScale(img, 1.1, 9)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face = img[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        if time % set_time == 0:
            image = Image.fromarray(np.uint8(face_resize)).convert('RGB')
            label, confidence = predict(image)

        cv2.putText(
            frame,
            str(label),
            (x+5, y-5),
            font,
            0.6,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            str(confidence),
            (x+5, y+h+23),
            font,
            0.7,
            (255, 255, 0),
            1
        )

    cv2.imshow('Emotion Regconizer v0.0', frame)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' or 'q' for exiting video
    if k == 27 or k == ord('q'):
        break
