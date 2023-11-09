import requests
import cv2
import base64

IMG_PATH = './elevate-nYgy58eb9aw-unsplash.jpg'
API_URL = 'REPLACE_WITH_ENDPOINT_URL'

# encode image to b64
with open(IMG_PATH, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')

# trigger api
result = requests.get(API_URL, json={"image": img_b64})

# extract detections
detections = result.json()['detections']
print(detections)

# display detections
img = cv2.imread(IMG_PATH)
for det in detections:
    x0,y0,x1,y1 = det['bbox']
    img = cv2.rectangle(img, (x0,y0), (x1,y1), (255,0,0), 4)
cv2.imwrite('./output.jpg', img)