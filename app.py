import streamlit as st
import mediapipe as mp
import streamlit_webrtc as webrtc
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class VideoTransformer(webrtc.VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return img

webrtc_ctx = webrtc.WebRtcMode.SENDRECV

webrtc_streamer = webrtc.webrtc_streamer(
    key="pose-estimation",
    mode=webrtc_ctx,
    video_transformer_factory=VideoTransformer,
)

if webrtc_streamer.is_recording():
    while True:
        video_frame = webrtc_streamer.get_frame()

        if video_frame is None:
            st.write("Finished")
            break

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(video_frame.to_ndarray(format="rgb24"))
            mp_drawing.draw_landmarks(
                video_frame.to_ndarray(format="rgb24"),
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )
        st.image(video_frame.to_ndarray(format="rgb24"), channels="RGB")
