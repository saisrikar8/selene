import os
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/ace/epam/level2_ascii/mfsa/Hourly/"
DEST_DIR = "../data/sep-data/epam-data-2"
CONCURRENCY = 32

os.makedirs(DEST_DIR, exist_ok=True)

async def fetch_file(session, filename, sem):
    url = BASE_URL + filename
    dest = os.path.join(DEST_DIR, filename)
    async with sem:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                async with aiofiles.open(dest, "wb") as f:
                    await f.write(await resp.read())
                return filename
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            return None

async def main():
    html = requests.get(BASE_URL).text
    soup = BeautifulSoup(html, "html.parser")
    filenames = [a['href'] for a in soup.find_all("a") if a['href'].endswith(".csv")]

    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_file(session, fname, sem) for fname in filenames]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await f

if __name__ == "__main__":
    import requests
    asyncio.run(main())
