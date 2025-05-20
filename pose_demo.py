import cv2
import mediapipe as mp

# Initialiseer MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Zet het beeld om naar RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Terug naar BGR voor OpenCV-weergave
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Teken de pose
    if results.pose_landmarks:
    
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    image = cv2.flip(image, 1)  # 1 = horizontaal spiegelen

    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Druk op ESC om te stoppen
        break

cap.release()
cv2.destroyAllWindows()
