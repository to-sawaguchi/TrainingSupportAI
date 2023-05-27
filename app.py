import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        st.image(img, channels="BGR")


def main():
    webrtc_ctx = webrtc_streamer(
        key="pose-estimation",
        mode=WebRtcMode.RECVONLY,
        video_transformer_factory=VideoTransformer,
        async_transform=True,
        device=1
    )

    if webrtc_ctx.video_transformer:
        st.write("カメラ映像を表示します")
        exit_button = st.button("終了", key="exit-button")
        while True:
            if exit_button:
                break


if __name__ == "__main__":
    main()
