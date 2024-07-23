import cv2
import numpy as np
import mediapipe as mp

# Define the keys and their positions on the virtual keyboard
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"]
]

# Define some constants
KEY_WIDTH = 60
KEY_HEIGHT = 60
GAP = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2

# Function to draw the keyboard
def draw_keyboard(img):
    x, y = GAP, GAP
    for row in keys:
        for key in row:
            cv2.rectangle(img, (x, y), (x + KEY_WIDTH, y + KEY_HEIGHT), (255, 255, 255), -1)
            cv2.putText(img, key, (x + 15, y + 40), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
            x += KEY_WIDTH + GAP
        x = GAP
        y += KEY_HEIGHT + GAP

# Function to detect which key is pressed
def get_key_pressed(x, y):
    row = (y - GAP) // (KEY_HEIGHT + GAP)
    col = (x - GAP) // (KEY_WIDTH + GAP)
    if row < len(keys) and col < len(keys[row]):
        return keys[row][col]
    return None

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the virtual keyboard window
keyboard_img = np.zeros((300, 700, 3), np.uint8)

# Draw the initial keyboard
draw_keyboard(keyboard_img)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the coordinates of the index fingertip
            index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = frame.shape
            cx, cy = int(index_fingertip.x * w), int(index_fingertip.y * h)

            # Detect if the index fingertip is pressing a key
            key = get_key_pressed(cx, cy)
            if key:
                cv2.rectangle(frame, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 255, 0), -1)
                print(f"Key pressed: {key}")

    cv2.imshow("Virtual Keyboard", keyboard_img)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
