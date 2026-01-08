import os
import json
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Load your image and set up your prompt
with open('sink.jpg', 'rb') as f:
    image_bytes = f.read()

points_data = []
prompt = """
        Clean this sink.
        The points should be labeled by order of the trajectory, from '0'
        (start point at left hand) to <n> (final point)
        The answer should follow the json format:
        [{"point": <point>, "label": <label1>}, ...].
        The points are in [y, x] format normalized to 0-1000.
        Give at least 20 points.
        """

image_response = client.models.generate_content(
  model="gemini-robotics-er-1.5-preview",
  contents=[
    types.Part.from_bytes(
      data=image_bytes,
      mime_type='image/jpeg',
    ),
    prompt
  ],
  config = types.GenerateContentConfig(
      temperature=0.5,
  )
)

print(image_response.text)

try:
    # Parse JSON from response (handling potential markdown code blocks)
    text = image_response.text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    points = json.loads(text)

    # Visualize points on the image
    with Image.open("sink.jpg") as image:
        draw = ImageDraw.Draw(image)
        width, height = image.size

        for item in points:
            y_norm, x_norm = item["point"]
            x = (x_norm / 1000) * width
            y = (y_norm / 1000) * height
            r = 5
            draw.ellipse((x - r, y - r, x + r, y + r), fill="red", outline="white")
            draw.text((x + 8, y - 8), item["label"], fill="red")

        image.save("trajectory_output.png")
        print("Saved visualization to trajectory_output.png")
except Exception as e:
    print(f"Error visualizing: {e}")
