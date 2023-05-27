import streamlit as st
import mediapipe as mp
import streamlit_webrtc as webrtc

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

webrtc_streamer = webrtc.Streamer(
    video_transformer_factory=None,
    bundle_errors=True,
)

if webrtc_streamer:
    video_frame = webrtc_streamer.recv()
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(video_frame.to_ndarray(format="rgb24"))
        mp_drawing.draw_landmarks(video_frame.to_ndarray(format="rgb24"), results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    st.image(video_frame.to_ndarray(format="rgb24"), channels="RGB")
