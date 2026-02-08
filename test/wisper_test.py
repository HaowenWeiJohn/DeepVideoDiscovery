import os
import whisper
from whisper.utils import get_writer

video_path = "/home/hwjwei/projects/longvideo/DeepVideoDiscovery/video_database/raw/nW8cWT2ufAQ.mp4"
out_dir = "/home/hwjwei/projects/longvideo/DeepVideoDiscovery/test/subs"
os.makedirs(out_dir, exist_ok=True)

model = whisper.load_model("large")  # "large" in openai-whisper is high quality
result = model.transcribe(
    video_path,
    task="transcribe",
    # If you know the language, set it for higher accuracy + speed:
    # language="en",
    verbose=True
)

# just print the result
print(result)

writer = get_writer("srt", out_dir)
writer(result, video_path)

# print("Wrote SRT to:", out_dir)
