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


PREFIX_PROMPT = (                                                                                                                                     
    "You are observing a video recording of an ADOS-2 (Autism Diagnostic Observation Schedule, 2nd Edition) "                                
    "clinical assessment session. In this video, a trained clinician administers standardized activities and "                                 
    "social presses to a child. Your task is to carefully watch the child's behavior throughout the session — "                                
    "including their language, social interactions, eye contact, gestures, play, and repetitive behaviors — "
    "and then score the following item based on what you observe. "
    "Focus on the child's behavior, not the clinician's.\n\n"
)


data_root = '/scratch/hwjwei/ADOS/data'
# list all the video files in the data_root
video_files = [f for f in os.listdir(data_root) if f.endswith('.mp4')]
print(video_files)

# print number of video files
print(f"Number of video files: {len(video_files)}")

results_folder = '/scratch/hwjwei/ADOS/results'
os.makedirs(results_folder, exist_ok=True)

for video_file in video_files:

    print("************************************************************************************")
    print(f"Analyzing video file: {video_file}")
    print("************************************************************************************")
    print()

    # video without .mp4 extension
    video_id = os.path.splitext(video_file)[0]
    raw_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, "raw")
    raw_path = os.path.join(raw_dir, f"{video_id}.mp4")
    video_folder = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id)
    frames_dir = os.path.join(video_folder, "frames")
    captions_dir = os.path.join(video_folder, "captions")
    video_db_path = os.path.join(video_folder, "database.json")
    srt_path = os.path.join(video_folder, "subtitles.srt")

    caption_file = os.path.join(captions_dir, "captions.json")
    # check if caption file exists, if not, warning
    if not os.path.exists(caption_file):
        print(f"Caption file {caption_file} not found")
        continue
    else:
        print(f"Caption file {caption_file} found")


    results_file = os.path.join(results_folder, video_file.replace('.mp4', '.json'))
    # create a json file if not exists
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            json.dump({}, f)
    
    # load the json file
    with open(results_file, 'r') as f:
        results = json.load(f)

    participant_id = video_file.split('_')[0]
    video_path = os.path.join(data_root, video_file)
    # check if the video file exists
    if not os.path.exists(video_path):
        print(f"Video file {video_file} not found")
        continue

    ados_scoring_sheet_name = 'ados_scoring_sheet_' + participant_id + '.csv'
    ados_scoring_sheet_path = os.path.join(data_root, ados_scoring_sheet_name)
    if not os.path.exists(ados_scoring_sheet_path):
        print(f"ADOS scoring sheet {ados_scoring_sheet_name} not found")
        continue
    ados_scoring_sheet = pd.read_csv(ados_scoring_sheet_path)
    print(ados_scoring_sheet.head())


    # iterate over the questions in the ados scoring sheet
    for idx, row in ados_scoring_sheet.iterrows():

        print("--------------------------------------------------------------------------------------")
        print(f"Processing question {idx}")
        print("--------------------------------------------------------------------------------------")
        print()
        # check if there is a index already in the results, idx is a string
        if idx in results:
            print(f"Question {idx} already processed")
            continue
        else:
            question = row.to_dict().copy()
            true_label_row_name = ados_scoring_sheet.columns[-1]
            ground_truth = row[true_label_row_name]
            question['ground_truth_module'] = ground_truth
            # add item dvd_module as a empty empty list
            question['dvd_module'] = []
            # add random_sample_module as a empty empty list
            question['random_sample_8_module'] = []
            

            # results[idx] = question
            results[str(idx)] = question
            # save the results to the json file
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
        
        queestion = results[str(idx)]
        
        prompt = (
            f"Module: {question['module']}\n"
            f"Test Type: {question['test_type']}\n"
            f"Description: {question['description']}\n\n"
            f"Provide ONE rating from the following options (ADOS codes):\n"
            f"{question['labels']}\n\n"
            f"Answer with a SINGLE integer code."
        )

        agent_prompt = PREFIX_PROMPT + prompt
        print(f"Agent prompt: {agent_prompt}")
        print("--------------------------------")
        print()
        
        # Run DVD Agent
        print("Initializing DVDCoreAgent...")
        agent = DVDCoreAgent(video_db_path, caption_file, config.MAX_ITERATIONS)
        print("Agent initialized.")

        print(f"Running agent with question: '{question}'")
        msgs = agent.run(agent_prompt)
        answer = extract_answer(msgs[-1])
        # print DVD agent answer and ground truth answer
        print(f"DVD Agent answer: {answer}")
        print(f"Ground truth answer: {ground_truth}")

        # add the answer to the dvd_module
        question['dvd_module'].append(answer)
        # save the results to the json file
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        print("--------------------------------")
        print(f"Question {idx} processed successfully")
        print("--------------------------------")
        print()

    print("--------------------------------")
    print(f"Video file: {video_file} processed successfully")
    print("--------------------------------")
    print()




    
    

































