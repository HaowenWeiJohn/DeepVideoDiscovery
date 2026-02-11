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
        if str(idx) in results:
            print(f"Question {idx} already processed")
            # continue
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
        
        question = results[str(idx)]
        
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

        print("Initializing DVDCoreAgent...")
        agent = DVDCoreAgent(video_db_path, caption_file, config.MAX_ITERATIONS)
        print("Agent initialized.")

        
        # Run DVD Agent and sampled images agent
        print("--------------------------------")
        print("Running DVD Agent...")
        print()
        dvd_agent_msgs = agent.run(agent_prompt)
        dvd_agent_answer = extract_answer(dvd_agent_msgs[-1])
        print(f"DVD Agent answer: {dvd_agent_answer}")
        question['dvd_module'].append(dvd_agent_answer)

        print("--------------------------------")
        print("Running with sampled images...")
        print()
        sampled_images_msgs = agent.run_with_sampled_images(agent_prompt)
        sampled_images_answer = extract_answer(sampled_images_msgs[-1])
        print(f"Sampled images answer: {sampled_images_answer}")
        question['random_sample_8_module'].append(sampled_images_answer)

        print("--------------------------------")
        print(f"Ground truth answer: {ground_truth}")

        # add the answer to the dvd_module


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




    
#     --------------------------------------------------------------------------------------
# Processing question 28
# --------------------------------------------------------------------------------------

# Agent prompt: You are observing a video recording of an ADOS-2 (Autism Diagnostic Observation Schedule, 2nd Edition) clinical assessment session. In this video, a trained clinician administers standardized activities and social presses to a child. Your task is to carefully watch the child's behavior throughout the session — including their language, social interactions, eye contact, gestures, play, and repetitive behaviors — and then score the following item based on what you observe. Focus on the child's behavior, not the clinician's.

# Module: B: Reciprocal Social Interaction
# Test Type: B18. Overall Quality of Rapport
# Description: The code for this item is a summary rating that reflects the examiner's overall judgment of the rapport or comfort level established with the child during the ADOS-2 evaluation. The rating should take into account the degree to which the examiner had to modify his or her own behavior to maintain the interaction successfully.

# Provide ONE rating from the following options (ADOS codes):
# 0 = Comfortable interaction between the child and examiner that is appropriate to the context of the ADOS-2 assessment.
# 1 = Interaction sometimes comfortable, but not sustained (e.g., sometimes feels awkward or stilted, or the child's behavior seems mechanical or slightly inappropriate).
# 2 = One-sided or unusual interaction resulting in a consistently mildly uncomfortable session.
# 3 = The child shows minimal regard for the examiner AND/OR the observation was markedly difficult or uncomfortable for a significant proportion of the time.

# Answer with a SINGLE integer code.
# Initializing DVDCoreAgent...
# INFO:nano-vectordb:Load (274, 2560) data
# INFO:nano-vectordb:Init {'embedding_dim': 2560, 'metric': 'cosine', 'storage_file': '/scratch/hwjwei/ADOS/video_dataset/deborah_module_t/database.json'} 274 data
# Database /scratch/hwjwei/ADOS/video_dataset/deborah_module_t/database.json already exists.
# Agent initialized.
# --------------------------------
# Running DVD Agent...

# Calling function `global_browse_tool` with args: {'database': NanoVectorDB(embedding_dim=2560, metric='cosine', storage_file='/scratch/hwjwei/ADOS/video_dataset/deborah_module_t/database.json'), 'query': 'Overall quality of rapport between child and examiner during ADOS-2 assessment'}
# Calling function `frame_inspect_tool` with args: {'database': NanoVectorDB(embedding_dim=2560, metric='cosine', storage_file='/scratch/hwjwei/ADOS/video_dataset/deborah_module_t/database.json'), 'question': 'Does the child show consistent comfort and reciprocal interaction with the examiner, or are there frequent signs of discomfort, avoidance, or one-sided interaction?', 'time_ranges_hhmmss': [['00:00:00, 00:45:39']]}
# Traceback (most recent call last):
#   File "/home/hwjwei/projects/longvideo/DeepVideoDiscovery/ados_benchmark_run.py", line 147, in <module>
#     dvd_agent_msgs = agent.run(agent_prompt)
#                      ^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/hwjwei/projects/longvideo/DeepVideoDiscovery/dvd/dvd_core.py", line 160, in run
#     self._exec_tool(tool_call, msgs)
#   File "/home/hwjwei/projects/longvideo/DeepVideoDiscovery/dvd/dvd_core.py", line 117, in _exec_tool
#     result = self.name_to_function_map[name](**args)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/hwjwei/projects/longvideo/DeepVideoDiscovery/dvd/build_database.py", line 34, in frame_inspect_tool
#     start_secs = convert_hhmmss_to_seconds(time_range[0])
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/hwjwei/projects/longvideo/DeepVideoDiscovery/dvd/build_database.py", line 235, in convert_hhmmss_to_seconds
#     hours, minutes, seconds = map(int, parts)
#     ^^^^^^^^^^^^^^^^^^^^^^^
ValueError: invalid literal for int() with base 10: '00, 00'

































