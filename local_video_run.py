import dvd.config as config
import os
import shutil
import whisper
from whisper.utils import get_writer
from dvd.dvd_core import DVDCoreAgent
from dvd.video_utils import decode_video_to_frames
from dvd.frame_caption import process_video, process_video_lite
from dvd.utils import extract_answer

# ---- User config ----
video_path = "/home/hwjwei/projects/longvideo/DeepVideoDiscovery/video_database/raw/nW8cWT2ufAQ.mp4"
question = "What is the color of the background when the man is talkinig?"


def main():
    # 1. Derive paths from video filename
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

    # 2. Copy video to raw folder (skip if exists)
    if not os.path.exists(raw_path):
        print(f"Copying video to {raw_path}...")
        shutil.copy2(video_path, raw_path)
        print("Video copied.")
    else:
        print(f"Video already exists at {raw_path}, skipping copy.")

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

    # 5. Run agent
    print("Initializing DVDCoreAgent...")
    agent = DVDCoreAgent(video_db_path, caption_file, config.MAX_ITERATIONS)
    print("Agent initialized.")

    print(f"Running agent with question: '{question}'")
    msgs = agent.run(question)
    print(extract_answer(msgs[-1]))


if __name__ == "__main__":
    main()
