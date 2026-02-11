
# AOAI_CAPTION_VLM_MODEL_NAME = 'Qwen/Qwen3-VL-235B-A22B-Instruct-FP8'
# AOAI_ORCHESTRATOR_LLM_MODEL_NAME = 'Qwen/Qwen3-VL-235B-A22B-Instruct-FP8'
# AOAI_TOOL_VLM_MODEL_NAME = 'Qwen/Qwen3-VL-235B-A22B-Instruct-FP8'
# AOAI_TOOL_VLM_MAX_FRAME_NUM = 50

# AOAI_EMBEDDING_LARGE_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
# AOAI_EMBEDDING_LARGE_DIM = 2560


import os
import json
import pandas as pd
import dvd.config as config
import os
import whisper
from whisper.utils import get_writer
from dvd.dvd_core import DVDCoreAgent
from dvd.video_utils import decode_video_to_frames
from dvd.frame_caption import process_video, process_video_lite
from dvd.utils import extract_answer



data_root = '/scratch/hwjwei/ADOS/data'
# list all the video files in the data_root
video_files = [f for f in os.listdir(data_root) if f.endswith('.mp4')]
print(video_files)

# print number of video files
print(f"Number of video files: {len(video_files)}")

for video_file in video_files:
    print("--------------------------------")
    print(f"Processing video file: {video_file}")
    print("--------------------------------")
    video_path = os.path.join(data_root, video_file)
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    raw_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, "raw")
    raw_path = os.path.join(raw_dir, f"{video_id}.mp4")
    video_folder = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id)
    frames_dir = os.path.join(video_folder, "frames")
    captions_dir = os.path.join(video_folder, "captions")
    video_db_path = os.path.join(video_folder, "database.json")
    srt_path = os.path.join(video_folder, "subtitles.srt")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(captions_dir, exist_ok=True)

    # 2. Symlink video to raw folder (skip if exists)
    if not os.path.exists(raw_path):
        print(f"Symlinking video to {raw_path}...")
        os.symlink(os.path.abspath(video_path), raw_path)
        print("Video symlinked.")
    else:
        print(f"Video already exists at {raw_path}, skipping.")

    # 3. Generate SRT with Whisper (skip if exists)
    if not os.path.exists(srt_path):
        print("Generating SRT with Whisper...")
        model = whisper.load_model("large")
        result = model.transcribe(raw_path, task="transcribe", verbose=True)
        writer = get_writer("srt", video_folder)
        writer(result, raw_path)
        # Whisper writes {video_id}.srt based on input filename; rename to subtitles.srt
        whisper_output = os.path.join(video_folder, f"{video_id}.srt")
        if os.path.exists(whisper_output) and whisper_output != srt_path:
            os.rename(whisper_output, srt_path)
        print(f"SRT generated at {srt_path}.")
    else:
        print(f"SRT already exists at {srt_path}, skipping generation.")

    # 4. Process based on LITE_MODE
    caption_file = os.path.join(captions_dir, "captions.json")

    if config.LITE_MODE:
        print("Running in LITE_MODE.")
        process_video_lite(captions_dir, srt_path)
    else:
        # Decode video to frames
        if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
            print(f"Decoding video to frames in {frames_dir}...")
            decode_video_to_frames(raw_path)
            print("Video decoded.")
        else:
            print(f"Frames already exist in {frames_dir}.")

        # Generate captions
        if not os.path.exists(caption_file):
            print("Processing video to get captions...")
            process_video(frames_dir, captions_dir, subtitle_file_path=srt_path)
            print("Captions generated.")
        else:
            print(f"Captions already exist at {caption_file}.")

    print("--------------------------------")
    print(f"Video file: {video_file} processed successfully")
    print("--------------------------------")
    print()