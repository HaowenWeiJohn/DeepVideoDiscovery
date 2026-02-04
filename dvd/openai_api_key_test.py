import dotenv
dotenv.load_dotenv()
import os
openai_api_key = os.getenv("OPENAI_API_KEY")
# test the api key
import openai
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)

