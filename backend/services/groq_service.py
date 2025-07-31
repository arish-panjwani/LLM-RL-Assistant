'''import aiohttp

async def call_groq(text: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.groq.com/endpoint", json={"text": text}) as resp:
            data = await resp.json()
            return data.get("result", text)

from backend.utils.config import settings
import aiohttp
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = settings.GROQ_API_KEY
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

async def call_groq(prompt: str):
    if not GROQ_API_KEY:
        return {"error": "Missing GROQ_API_KEY"}

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model":"llama2-70b-4096", #"mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(GROQ_API_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                return {"error": f"Groq API call failed: {resp.status}"}
            result = await resp.json()
            return result["choices"][0]["message"]["content"]
'''      

from backend.utils.config import settings
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
# IMPORTANT: correct endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

async def call_groq(prompt: str):
    if not GROQ_API_KEY:
        return {"error": "Missing GROQ_API_KEY"}

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(GROQ_API_URL, headers=headers, json=payload) as resp:
            text = await resp.text()
            print("Groq raw response:", text)

            if resp.status != 200:
                return {"error": f"Groq API call failed: {resp.status}"}

            result = await resp.json()
            return result["choices"][0]["message"]["content"]

