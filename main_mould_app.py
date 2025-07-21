import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
from ultralytics import YOLO
import numpy as np

# ─── 1. Load YOLO Model ───────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\ayawa\PycharmProjects\PythonProject7\Colab Notebooks\Data\runs\detect\mould_with_negatives\weights\best.pt"
CONF_THRESH = 0.5
IOU_THRESH = 0.45

model = YOLO(MODEL_PATH)

# ─── 2. Define Video Transformer ──────────────────────────────────────────────────
class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        results = model.predict(image, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = float(box.conf[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(
                    image,
                    f"{conf_score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2
                )

        return image

# ─── 3. Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Mould Detector", layout="centered")
st.title("🦠 Live Mould Detection with YOLOv8")

st.info("Click the button below to start the webcam and detect mould in real time.")

webrtc_streamer(
    key="yolo-stream",
    video_transformer_factory=YOLOVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
