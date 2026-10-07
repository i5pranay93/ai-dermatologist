import base64
import os
import subprocess
import tempfile
from io import BytesIO

from dotenv import load_dotenv
from groq import Groq
from PIL import Image


load_dotenv()


def encode_image_for_groq(filepath):
    image = Image.open(filepath)
    image.thumbnail((1024, 1024))

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_frames_for_groq(video_filepath, num_frames=3):
    """Sample up to num_frames JPEGs from a video with ffmpeg.

    Helper for the commented-out video path in brain_of_the_doctor below.
    Returns a list of temporary file paths. Tested working with Groq vision.
    """
    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_filepath),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError as error:
        raise ValueError(f"Could not read video duration: {video_filepath}") from error
    if duration <= 0:
        raise ValueError(f"Video has no usable duration: {video_filepath}")

    timestamps = [duration * (i + 1) / (num_frames + 1) for i in range(num_frames)]
    frame_dir = tempfile.mkdtemp(prefix="derm_frames_")
    frames = []
    for index, timestamp in enumerate(timestamps):
        frame_path = os.path.join(frame_dir, f"frame_{index}.jpg")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss", str(timestamp),
                "-i", str(video_filepath),
                "-frames:v", "1",
                "-q:v", "3",
                frame_path,
            ],
            capture_output=True,
            check=True,
        )
        frames.append(frame_path)
    return frames


def brain_of_the_doctor(patient_text, image_filepath=None, video_filepath=None):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in .env or environment")

    # Active path: free Groq vision, image only. Groq accepts up to 3 images
    # per request, never video.
    #
    # FORKERS: real video support needs a model with native video input, for
    # example MiniMax. Until you have such an API key, the free stopgap below
    # samples frames from the video with ffmpeg and sends them as images.
    # It is tested working: uncomment it, restore video_filepath in the UI
    # (main.py), and remove the image-only guard.
    #
    # video_frames = []
    # if video_filepath:
    #     video_frames = extract_frames_for_groq(video_filepath, num_frames=3)
    #
    # if image_filepath:
    #     visual_paths = [image_filepath] + video_frames[:2]
    #     visual_note = "\nUse the attached skin image and any attached video frames as the visual reference."
    # elif video_frames:
    #     visual_paths = video_frames[:3]
    #     visual_note = "\nThe patient uploaded a video instead of a still image. Review the attached frames sampled from it."
    # else:
    #     raise ValueError("Please upload a skin image or video.")

    if not image_filepath:
        raise ValueError("Groq vision requires an image. Please upload a skin image.")

    visual_paths = [image_filepath]
    visual_note = "\nUse the attached skin image as the visual reference."

    prompt = (
        "You are a confident, natural doctor specializing in skin care. Speak with the reassurance, clarity, and authority of a real doctor. "
        "Limit your entire response to two or three sentences maximum. "
        "Do not use any special characters, symbols, asterisks, or markdown formatting in your response because it will be converted directly to audio.\n\n"
        f"Patient text: {patient_text}"
    )
    prompt += visual_note

    content = [{"type": "text", "text": prompt}]
    for path in visual_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_for_groq(path)}",
                },
            }
        )

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
        max_completion_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": "You are a careful skin care assistant. Give general information, not a diagnosis.",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    return response.choices[0].message.content


# OLD CODE KEPT FOR REFERENCE
# import base64
# import os
# from io import BytesIO
#
# from dotenv import load_dotenv
# from groq import Groq
# from PIL import Image
#
#
# folder = os.path.dirname(__file__)
# env_path = os.path.join(folder, ".env")
# load_dotenv(env_path)
#
# api_key = os.environ.get("GROQ_API_KEY")
# if not api_key:
#     raise ValueError("Missing GROQ_API_KEY in .env or environment")
#
#
# image_path = os.path.join(folder, "sample-image.png")
#
# image = Image.open(image_path)
# image.thumbnail((1024, 1024))
#
# buffer = BytesIO()
# image.convert("RGB").save(buffer, format="JPEG", quality=75)
# image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
#
# client = Groq(api_key=api_key)
#
# response = client.chat.completions.create(
#     model=os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
#     max_completion_tokens=1000,
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful medical assistant. Give general information, not a diagnosis.",
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "What do you see in this image? Give general skin care advice, not a diagnosis.",
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{image_data}",
#                     },
#                 },
#             ],
#         },
#     ],
# )
#
# print(response.choices[0].message.content)
