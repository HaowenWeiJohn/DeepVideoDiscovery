import os
import dotenv

# ------------------ video download and segmentation configuration ------------------ #
VIDEO_DATABASE_FOLDER = "./video_database/"

# if the system is a linux machine, the Database folder is: /home/hwjwei/projects/longvideo/DeepVideoDiscovery/video_database/
VIDEO_DATABASE_FOLDER = "/home/hwjwei/projects/longvideo/DeepVideoDiscovery/video_database/" if os.name == "posix" else "./video_database/"

VIDEO_RESOLUTION = "360" # denotes the height of the video 
VIDEO_FPS = 2 # frames per second
CLIP_SECS = 10 # seconds

# ------------------ model configuration ------------------ #
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None) # will overwrite Azure OpenAI setting

# this will load the .env file and set the OPENAI_API_KEY environment variable
dotenv.load_dotenv()



SERVER = 'VLLM'

# server must be OPENAI or VLLM
assert SERVER in ["OPENAI", "VLLM"], "SERVER must be OPENAI or VLLM"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")


if SERVER == "OPENAI":
    print("OPENAI server is used")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    assert OPENAI_API_KEY is not None, "OPENAI_API_KEY is not set"
    ENDPOINT = 'https://api.openai.com/v1'
    AOAI_CAPTION_VLM_MODEL_NAME = "gpt-4.1-mini"
    AOAI_ORCHESTRATOR_LLM_MODEL_NAME = "o3"
    AOAI_TOOL_VLM_MODEL_NAME = "gpt-4.1-mini"
    AOAI_TOOL_VLM_MAX_FRAME_NUM = 50
    
elif SERVER == "VLLM":
    print("VLLM server is used")
    OPENAI_API_KEY = os.getenv("VLLM_API_KEY")
    assert OPENAI_API_KEY is not None, "VLLM_API_KEY is not set"
    ENDPOINT = 'https://api.dd.works/v1'
    AOAI_CAPTION_VLM_MODEL_NAME = 'Qwen/Qwen3-VL-235B-A22B-Instruct-FP8'
    AOAI_ORCHESTRATOR_LLM_MODEL_NAME = 'Qwen/Qwen3-VL-235B-A22B-Instruct-FP8'
    AOAI_TOOL_VLM_MODEL_NAME = 'Qwen/Qwen3-VL-235B-A22B-Instruct-FP8'
    AOAI_TOOL_VLM_MAX_FRAME_NUM = 50


if SERVER == "OPENAI":
    EMBEDDING_ENDPOINT = "https://api.openai.com/v1"
    AOAI_EMBEDDING_LARGE_MODEL_NAME = "text-embedding-3-large"
    AOAI_EMBEDDING_LARGE_DIM = 3072
elif SERVER == "VLLM":
    EMBEDDING_ENDPOINT = "http://localhost:8888/v1"
    AOAI_EMBEDDING_LARGE_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
    AOAI_EMBEDDING_LARGE_DIM = 2560




AOAI_CAPTION_VLM_ENDPOINT_LIST = [""]
# AOAI_CAPTION_VLM_MODEL_NAME = "gpt-4.1-mini"

AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST = [""]
# AOAI_ORCHESTRATOR_LLM_MODEL_NAME = "o3"

AOAI_TOOL_VLM_ENDPOINT_LIST = [""]
# AOAI_TOOL_VLM_MODEL_NAME = "gpt-4.1-mini"
# AOAI_TOOL_VLM_MAX_FRAME_NUM = 50

AOAI_EMBEDDING_RESOURCE_LIST = [""]

# ------------------ agent and tool setting ------------------ #
LITE_MODE = False # if True, only leverage srt subtitle, no pixel downloaded or pixel captioning
GLOBAL_BROWSE_TOPK = 300
OVERWRITE_CLIP_SEARCH_TOPK = 0 # 0 means no overwrite and let agent decide

SINGLE_CHOICE_QA = True  # Design for benchmark test. If True, the agent will only return options for single-choice questions.
MAX_ITERATIONS = 10  # Maximum number of iterations for the agent to run