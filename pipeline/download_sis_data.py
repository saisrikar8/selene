import os
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/ace/sis/level_2_cdaweb/sis_h1/"
DEST_DIR = "../data/sep-data/sis-data"
START_YEAR = 1997
END_YEAR = 2024
CONCURRENT_REQUESTS = 32

os.makedirs(DEST_DIR, exist_ok=True)

async def fetch_html(session, url, sem):
    async with sem:
        async with session.get(url, timeout=60) as response:
            response.raise_for_status()
            return await response.text()

async def get_cdf_links_for_year(session, year, sem):
    year_url = f"{BASE_URL}{year}/"
    try:
        html = await fetch_html(session, year_url, sem)
        soup = BeautifulSoup(html, "html.parser")
        links = [a['href'] for a in soup.find_all("a", href=True) if a['href'].endswith(".cdf")]
        return [(f"{year_url}{link}", os.path.join(DEST_DIR, str(year), link)) for link in links]
    except Exception as e:
        print(f"[{year}] Failed to fetch list: {e}")
        return []

async def download_file(session, url, dest, sem):
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        async with sem:
            async with session.get(url, timeout=60) as response:
                response.raise_for_status()
                async with aiofiles.open(dest, 'wb') as f:
                    await f.write(await response.read())
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")

async def main():
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)  # ✅ create within same loop
    all_tasks = []

    async with aiohttp.ClientSession() as session:
        all_links = []
        for year in range(START_YEAR, END_YEAR + 1):
            print(f"Fetching links for {year}...")
            links = await get_cdf_links_for_year(session, year, sem)
            all_links.extend(links)

        print(f"Downloading {len(all_links)} files...")

        tasks = [download_file(session, url, dest, sem) for url, dest in all_links]
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            await coro

if __name__ == "__main__":
    asyncio.run(main())
