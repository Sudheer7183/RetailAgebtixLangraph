# utils/ollama_client.py
import requests
import json
from typing import Dict, Any

OLLAMA_URL = "http://localhost:11434/api/generate"  # adjust if different
OLLAMA_MODEL = "llama3.2:3b"  # change if necessary

def call_ollama(prompt: str, max_tokens: int = 512, stream: bool = False) -> Dict[str, Any]:
    """
    Call Ollama /api/generate with a simple prompt and return parsed text.
    """
    # payload = {
    #     "model": OLLAMA_MODEL,
    #     "prompt": prompt,
    #     "max_tokens": max_tokens,
    #     "stream": stream
    # }
    # resp = requests.post(OLLAMA_URL, json=payload)
    # resp=requests.post(
    #                 "https://api.gemini.cloud/v1/generate",  # Example URL, replace with actual Gemini API endpoint
    #                 headers={
    #                     "Authorization": "Bearer ************************",  # Add your Gemini API key
    #                     "Content-Type": "application/json"
    #                 },
    #                 json={
    #                     "model": "google_genai:gemini-2.0-flash",  # Specify the model name as per Gemini Cloud's documentation
    #                     "prompt": prompt,
    #                     "max_tokens": 1000,  # You can adjust the max tokens as needed
    #                     "temperature": 0.7,  # Adjust temperature for output variability
    #                     "stream": False
    #                 }
    #             )
    # api_key="AIzaSyCgA0OLuMrqXUvaM3pXBl-ohTe0j5pqahQ"
    api_key="AIzaSyB0dexOy6Vmi3cU_490RwddvFJr8XITUaA"
    model_name="gemini-2.0-flash"
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json",
                # possibly other headers if required
    }
    body={
        "contents":[
            {
                "parts":[
                    {"text":prompt}
                ]
            }
        ],
            "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1000
        }
    }

    resp=requests.post(
        url,
        headers=headers,
        json=body
    )


    resp.raise_for_status()
    # Ollama returns streaming chunks; basic non-stream usage returns JSON with text in 'results'
    data = resp.json()
    # try to find text
    text = ""
    if isinstance(data, dict):
        # streaming may put pieces in data["results"]
        if "results" in data and isinstance(data["results"], list):
            # find first result with "output"
            for r in data["results"]:
                if isinstance(r, dict) and "content" in r:
                    # content can be list of dicts, find text fragments
                    for c in r["content"]:
                        if c.get("type") == "output_text":
                            text += c.get("text", "")
        elif "output" in data and isinstance(data["output"], list):
            # older style
            for out in data["output"]:
                if out.get("type") == "output_text":
                    text += out.get("text", "")
    # fallback raw
    if text == "":
        text = json.dumps(data)
    return {"raw": data, "text": text}
