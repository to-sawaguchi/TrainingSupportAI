import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# カウンター変数とフラグの初期化
counter = 0
stage = None
timer = None

def initialize_pose():
    return mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def initialize_camera():
    return cv2.VideoCapture(1)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_frame(camera):
    global counter, stage, timer  # グローバル変数を使用することを宣言
    success, image = camera.read()
    if success and image is not None:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    counter += 1
                    timer = time.time()
            except:
                pass

            if timer is not None and time.time() - timer > 6:
                return None, counter, stage

            cv2.putText(image, 'REPS: ' + str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE: ' + str(stage),
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # 関節の描画
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

    return image, counter, stage

def main():
    pose = initialize_pose()
    camera = initialize_camera()

    st.title("Reps Counter")
    st.write("カメラ映像を表示します")
    image_slot = st.empty()
    counter_slot = st.empty()
    stage_slot = st.empty()

    while True:
        frame, counter, stage = get_frame(camera)

        # 終了条件
        if frame is None:
            st.write("Finished")
            break

        # 画像を表示
        image_slot.image(frame, channels="BGR")
        counter_slot.text(f"REPS: {counter}")
        stage_slot.text(f"STAGE: {stage}")

    # カメラをリリース
    camera.release()

if __name__ == "__main__":
    main()