from pydantic import BaseModel
from dotenv import load_dotenv
from google.genai import types
from google import genai
import os


directory_path = "output"


def extract_files_path(dir: str) -> list:
    list_of_files = []
    for dirpath, _, files_name in os.walk(directory_path):
        for file in files_name:
            file_path = os.path.join(dirpath, file)
            list_of_files.append(file_path)

    return list_of_files


with open(extract_files_path(directory_path)[0], encoding="utf-8") as file:
    import json
    data = json.load(file)

text = """"""
for i in range(4):
    text += list(data.keys())[i] + "\n" + data[list(data.keys())[i]] + "\n\n"


load_dotenv()

# Configure the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define the grounding tool
grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

model_name = 'gemini-2.5-flash'


class MatchResult(BaseModel):
    market_overview: str
    market_driver: str
    market_challenge: str
    market_trends: str


def rephrase_context(context: str):

    response = client.models.generate_content(
        model=model_name,
        contents=f"""Rephrase the following paragraph while preserving its original meaning, tone, and intent. Do not add new information, remove details, or change the context. Improve clarity and flow only where necessary, keeping the message accurate and faithful to the original text.
        
        # Context
        {context}
        
        """,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": MatchResult.model_json_schema(),
        }
    )
    result = MatchResult.model_validate_json(response.text)

    return result.model_dump()


response = rephrase_context(text)

print(response)


def rephrase_response_keys(data: dict):
    data['Market Overview'] = data['market_overview']
    data['Market Driver'] = data['market_driver']
    data['Market Challenge'] = data['market_challenge']
    data['Market Trends'] = data['market_trends']

    del data['market_overview']
    del data['market_driver']
    del data['market_challenge']
    del data['market_trends']

    return data


response = rephrase_response_keys(response)

with open("output1.json", "w", encoding="utf-8") as f:
    json.dump(response, f, indent=2, ensure_ascii=False)
