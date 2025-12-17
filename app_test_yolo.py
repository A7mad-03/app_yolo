import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="YOLOv8 Person Detection", layout="wide")

st.title("🎯 YOLOv8 Person Detection (Camera)")

# Load model ONCE
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Video processing class
class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # YOLOv8 inference
        results = model.predict(
            source=img,
            conf=0.3,
            classes=[0],  # person only
            verbose=False
        )

        # Draw results
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"Person {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        return img

# Start webcam stream
webrtc_streamer(
    key="yolo-person",
    video_transformer_factory=YOLOTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
