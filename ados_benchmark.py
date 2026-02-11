
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


video_name = 'luke_module_t.mp4'

participant_id = video_name.split('_')[0]

video_path = os.path.join(data_root, video_name)
# check existance of video
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file {video_name} not found")



ados_scoring_sheet_name = 'ados_scoring_sheet_' + participant_id + '.csv'
ados_scoring_sheet_path = os.path.join(data_root, ados_scoring_sheet_name)
if not os.path.exists(ados_scoring_sheet_path):
    raise FileNotFoundError(f"ADOS scoring sheet {ados_scoring_sheet_name} not found")

# read ados scoring sheet
ados_scoring_sheet = pd.read_csv(ados_scoring_sheet_path)
print(ados_scoring_sheet.head())


for idx, row in ados_scoring_sheet.iterrows():


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


    prefix = (                                                                                                                                     
      "You are observing a video recording of an ADOS-2 (Autism Diagnostic Observation Schedule, 2nd Edition) "                                
      "clinical assessment session. In this video, a trained clinician administers standardized activities and "                                 
      "social presses to a child. Your task is to carefully watch the child's behavior throughout the session — "                                
      "including their language, social interactions, eye contact, gestures, play, and repetitive behaviors — "
      "and then score the following item based on what you observe. "
      "Focus on the child's behavior, not the clinician's.\n\n"
    )

    prompt = (
        f"Module: {row['module']}\n"
        f"Test Type: {row['test_type']}\n"
        f"Description: {row['description']}\n\n"
        f"Provide ONE rating from the following options (ADOS codes):\n"
        f"{row['labels']}\n\n"
        f"Answer with a SINGLE integer code."
    )
    question = prefix + prompt


    # the last column name of the ados scoring sheet
    true_label_row_name = ados_scoring_sheet.columns[-1]

    true_label = row[true_label_row_name]
    print(f"--- Item {idx + 1} ---")

    print(f"Question: {question}")
    print(f"True label: {true_label}")
    print()

    # 5. Run agent
    print("Initializing DVDCoreAgent...")
    agent = DVDCoreAgent(video_db_path, caption_file, config.MAX_ITERATIONS)
    print("Agent initialized.")

    print(f"Running agent with question: '{question}'")
    msgs = agent.run(question)
    print(extract_answer(msgs[-1]))


















