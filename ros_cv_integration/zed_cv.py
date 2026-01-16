import cv2
import os
import json
from io import BytesIO
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
from google.genai.errors import ServerError
import re

import pyzed.sl as sl
import numpy as np

def generate_with_retry(client, **kwargs):
    retries = 5
    delay = 2  # seconds

    for attempt in range(retries):
        try:
            return client.models.generate_content(**kwargs)
        except ServerError as e:
            if e.status_code == 503:
                print(f"[WARN] Model overloaded, retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e

    raise RuntimeError("Model unavailable after retries")

# ---------------------------
# Load API key
# ---------------------------
load_dotenv()
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# ---------------------------
# ZED Camera Initialization
# ---------------------------
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.AUTO
init_params.camera_fps = 30

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("ZED open error:", err)
    exit()

runtime_params = sl.RuntimeParameters()
zed_image = sl.Mat()

print("Press 'c' to capture ZED frame and send to Gemini")
print("Press 'q' to quit")

captured_frame = None

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(zed_image, sl.VIEW.LEFT)

        # Convert sl.Mat → NumPy (RGBA)
        frame_rgba = zed_image.get_data()

        # Convert RGBA → RGB
        frame_rgb = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)

        cv2.imshow("ZED Left Image", frame_rgb)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            captured_frame = frame_rgb.copy()
            print("ZED frame captured")
            break
        elif key == ord('q'):
            zed.close()
            cv2.destroyAllWindows()
            exit()

zed.close()
cv2.destroyAllWindows()


# ---------------------------
# Convert OpenCV frame → JPEG bytes
# ---------------------------
# OpenCV uses BGR, PIL expects RGB
captured_frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(captured_frame_rgb)

buffer = BytesIO()
pil_image.save(buffer, format="JPEG")
image_bytes = buffer.getvalue()

# ---------------------------
# Prompt
# ---------------------------
prompt = """
Give the four corners of the box holding.
The answer should follow the json format:
[{"point": [y, x], "label": <label>}].
The points are in [y, x] format normalized to 0-1000.
"""

# ---------------------------
# Send to Gemini Robotics Model
# ---------------------------
response = generate_with_retry(
    client,
    model="gemini-robotics-er-1.5-preview",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        ),
        prompt
    ],
    config=types.GenerateContentConfig(
        temperature=0.5
    )
)


print("Model response:")
print(response.text)

# ---------------------------
# Parse JSON safely
# ---------------------------
try:
    text = response.text.strip()

    # Remove ```json or ``` safely
    text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()

    points = json.loads(text)

    # ---------------------------
    # Visualize points
    # ---------------------------
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size

    for item in points:
        y_norm, x_norm = item["point"]
        label = item.get("label", "")

        x = (x_norm / 1000) * width
        y = (y_norm / 1000) * height

        r = 6
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill="red",
            outline="white"
        )
        draw.text((x + 8, y - 8), label, fill="red")

    pil_image.save("trajectory_output.png")
    print("Saved trajectory_output.png")

except Exception as e:
    print("Error parsing or visualizing response:", e)
