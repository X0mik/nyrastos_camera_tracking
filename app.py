import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

landmarker = vision.HandLandmarker.create_from_model_path("hand_landmarker.task")

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),        # thumb
               (0,5),(5,6),(6,7),(7,8),        # index
               (5,9),(9,10),(10,11),(11,12),   # middle
               (9,13),(13,14),(14,15),(15,16), # ring
               (13,17),(0,17),(17,18),(18,19),(19,20)]  # pinky + palm

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    for hand in result.hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in hand]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow("hands", frame)
    cv2.waitKey(1)