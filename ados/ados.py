import os
import json
import pandas as pd



data_root = '/scratch/hwjwei/ADOS/data'
participant_id = 'luke'

video_name =  participant_id + '_module_t.mp4'
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
    prompt = (
        f"Module: {row['module']}\n"
        f"Test Type: {row['test_type']}\n"
        f"Description: {row['description']}\n\n"
        f"Provide ONE rating from the following options (ADOS codes):\n"
        f"{row['labels']}\n\n"
        f"Answer with a SINGLE integer code."
        #f"Based only on the behaviors described, assign the correct ADOS code for this item."
    )
    true_label = row[participant_id + '_module_t']
    print(f"--- Item {idx + 1} ---")
    true_label_string = f"True label: {true_label}"
    print(prompt)
    print()
    break  










