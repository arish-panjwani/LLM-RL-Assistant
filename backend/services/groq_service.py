import aiohttp

async def call_groq(text: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.groq.com/endpoint", json={"text": text}) as resp:
            data = await resp.json()
            return data.get("result", text)
