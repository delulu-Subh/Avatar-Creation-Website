import cv2
import mediapipe as mp
import numpy as np

# ==========================
# COLORS
# ==========================
CYAN = (255, 255, 0)
ORANGE = (0, 180, 255)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
CORE = (0, 255, 180)

# ==========================
# HAND TRACKING CLASS
# ==========================
class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )

    def get_landmarks(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        landmarks_list = []
        if results.multi_hand_landmarks:
            h, w, _ = img.shape
            for handLms in results.multi_hand_landmarks:
                lm = [(int(l.x * w), int(l.y * h)) for l in handLms.landmark]
                landmarks_list.append(lm)
        return landmarks_list

    def draw_hand_skeleton(self, img, hand_landmarks):
        self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

# ==========================
# AR HUD DRAWING FUNCTIONS
# ==========================
def draw_glow_circle(img, center, radius, color, thickness=2, glow=15):
    for g in range(glow, 0, -3):
        alpha = 0.08 + 0.12 * (g / glow)
        overlay = img.copy()
        cv2.circle(overlay, center, radius+g, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.circle(img, center, radius, color, thickness)

def draw_radial_ticks(img, center, radius, color, num_ticks=24, length=22, thickness=3):
    for i in range(num_ticks):
        angle = np.deg2rad(i * (360/num_ticks))
        x1 = int(center[0] + (radius-length) * np.cos(angle))
        y1 = int(center[1] + (radius-length) * np.sin(angle))
        x2 = int(center[0] + radius * np.cos(angle))
        y2 = int(center[1] + radius * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_core_pattern(img, center, radius):
    for t in np.linspace(0, 2*np.pi, 40):
        r = radius * (0.7 + 0.3 * np.sin(6*t))
        x = int(center[0] + r * np.cos(t))
        y = int(center[1] + r * np.sin(t))
        cv2.circle(img, (x, y), 3, ORANGE, -1)
    cv2.circle(img, center, int(radius*0.6), CYAN, 2)
    cv2.circle(img, center, int(radius*0.4), ORANGE, 2)

def draw_hud_details(img, center):
    for i in range(8):
        angle = np.deg2rad(210 + i*10)
        x1 = int(center[0] + 140 * np.cos(angle))
        y1 = int(center[1] + 140 * np.sin(angle))
        x2 = int(center[0] + 170 * np.cos(angle))
        y2 = int(center[1] + 170 * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), CYAN, 4)
    for i in range(4):
        angle = np.deg2rad(270 + i*15)
        x = int(center[0] + 120 * np.cos(angle))
        y = int(center[1] + 120 * np.sin(angle))
        cv2.rectangle(img, (x-10, y-10), (x+10, y+10), CYAN, 2)

def draw_arc_segments(img, center):
    cv2.ellipse(img, center, (110,110), 0, -30, 210, CYAN, 3)
    cv2.ellipse(img, center, (100,100), 0, -30, 210, ORANGE, 2)
    cv2.ellipse(img, center, (80,80), 0, 0, 360, CYAN, 1)

# ==========================
# GESTURE DETECTION
# ==========================
def detect_gesture(lm):
    palm = lm[9]
    tips = [lm[i] for i in [4, 8, 12, 16, 20]]
    dists = [np.linalg.norm(np.array(tip) - np.array(palm)) for tip in tips]
    avg_dist = np.mean(dists)
    pinch_dist = np.linalg.norm(np.array(lm[4]) - np.array(lm[8]))
    pinch_val = int(100 - min(pinch_dist, 100))
    if avg_dist > 70:
        return 'OPEN', palm, pinch_val
    elif pinch_val < 60:
        return 'PINCH', palm, pinch_val
    else:
        return 'FIST', palm, pinch_val

# ==========================
# MAIN LOOP
# ==========================
def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        landmarks_list = tracker.get_landmarks(frame)

        for lm in landmarks_list:
            gesture, palm, pinch_val = detect_gesture(lm)

            # Draw hand skeleton
            # tracker.draw_hand_skeleton(frame, hand_landmarks) # optional

            if gesture == 'OPEN':
                draw_glow_circle(frame, palm, 120, CYAN, 3, glow=30)
                draw_glow_circle(frame, palm, 90, CYAN, 2, glow=20)
                draw_glow_circle(frame, palm, 60, ORANGE, 2, glow=10)
                draw_radial_ticks(frame, palm, 120, CYAN)
                draw_core_pattern(frame, palm, 35)
                draw_hud_details(frame, palm)
                draw_arc_segments(frame, palm)
                for i in [4, 8, 12, 16, 20]:
                    cv2.line(frame, palm, lm[i], CYAN, 2)
                    cv2.circle(frame, lm[i], 12, ORANGE, -1)
                # Angle overlay
                v1 = np.array(lm[4]) - np.array(palm)
                v2 = np.array(lm[8]) - np.array(palm)
                cos_angle = np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6), -1.0, 1.0)
                angle = int(np.degrees(np.arccos(cos_angle)))
                cv2.putText(frame, f'{angle}°', (palm[0]+40, palm[1]-40),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, WHITE, 4)
            elif gesture == 'PINCH':
                draw_glow_circle(frame, palm, 60, ORANGE, 3, glow=20)
                cv2.putText(frame, f'Pinch: {pinch_val}', (palm[0]-40, palm[1]-70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 3)
                for i in range(5):
                    cv2.ellipse(frame, (palm[0]+80, palm[1]), (30,30),
                                0, 180, 180+pinch_val+i*10, ORANGE, 2)
            else:  # FIST
                draw_glow_circle(frame, palm, 60, CYAN, 3, glow=20)
                cv2.putText(frame, 'FIST', (palm[0]-30, palm[1]-70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 3)

        cv2.imshow('Hand Tracking AR UI', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
