import os
import requests
import dotenv
dotenv.load_dotenv()



BASE_URL = "https://api.dd.works/v1"
API_KEY = 'Wf9i1OQCgKA4nAVZ__8CYJhfhExt_Yob120jbjIz0yA'

MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"

if not API_KEY:
    raise SystemExit("Set VLLM_API_KEY first (your endpoint API key).")

headers = {"Authorization": f"Bearer {API_KEY}"}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Think about why the sky is blue."},
    ],
    "temperature": 0,
    "max_tokens": 100,
}


r = requests.post(
    f"{BASE_URL}/chat/completions",
    headers=headers,
    json=payload,
    timeout=60,
)

print("status:", r.status_code)
print(r.text)  # raw response for debugging
r.raise_for_status()

data = r.json()
print("assistant:", data["choices"][0]["message"]["content"])



# import os
# import requests
# import dotenv
# dotenv.load_dotenv()

# BASE_URL = "https://api.dd.works/v1"
# API_KEY = os.getenv("VLLM_API_KEY")
# MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"

# if not API_KEY:
#     raise SystemExit("Set VLLM_API_KEY first (your endpoint API key).")

# headers = {"Authorization": f"Bearer {API_KEY}"}

# payload = {
#     "model": MODEL,
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Think about why the sky is blue."},
#     ],
#     "temperature": 0.6,
#     "top_p": 0.95,
#     "max_tokens": 4096,
#     "chat_template_kwargs": {"enable_thinking": True},
# }

# r = requests.post(
#     f"{BASE_URL}/chat/completions",
#     headers=headers,
#     json=payload,
#     timeout=120,
# )

# print("status:", r.status_code)
# r.raise_for_status()

# data = r.json()
# msg = data["choices"][0]["message"]

# # If server has --enable-reasoning, thinking is separate
# if msg.get("reasoning_content"):
#     print("thinking:", msg["reasoning_content"])
# print("assistant:", msg["content"])
