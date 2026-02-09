from openai import OpenAI
import dotenv
import os
dotenv.load_dotenv()

API_KEY = os.getenv("VLLM_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="http://localhost:8888/v1",
)

response = client.embeddings.create(
    input=["Our evaluation indicates that, for most downstream tasks, using instructions (instruct) typically yields an improvement of 1% to 5% compared to not using them. Therefore, we recommend that developers create tailored instructions specific to their tasks and scenarios. In multilingual contexts, we also advise users to write their instructions in English, as most instructions utilized during the model training process were originally written in English."],
    model="Qwen/Qwen3-Embedding-4B",
)

for item in response.data:
    print(item.index, len(item.embedding))
