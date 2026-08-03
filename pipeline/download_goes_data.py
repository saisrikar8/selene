import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = 'https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/sgps-l2-avg1m//'

YEARS_MONTHS = {
    '2020': ['11', '12'],
    '2021': [f'{m:02d}' for m in range(1, 13)],
    '2022': [f'{m:02d}' for m in range(1, 13)],
    '2023': [f'{m:02d}' for m in range(1, 13)],
    '2024': [f'{m:02d}' for m in range(1, 13)],
    '2025': ['01', '02', '03', '04']
}

DOWNLOAD_DIR = './goes_seiss_data/'

def get_file_links(month_url):
    r = requests.get(month_url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    links = []
    for a in soup.find_all('a'):
        href = a.get('href')
        if href and (href.endswith('.cdf') or href.endswith('.csv') or href.endswith('.txt') or href.endswith('.nc')):
            links.append(urljoin(month_url, href))
    return links

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"Already downloaded: {save_path}")
        return
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    for year, months in YEARS_MONTHS.items():
        for month in months:
            month_url = urljoin(BASE_URL, f'{year}/{month}/')
            month_dir = os.path.join(DOWNLOAD_DIR, year, month)
            os.makedirs(month_dir, exist_ok=True)

            print(f"Listing files for {year}-{month} ...")
            try:
                files = get_file_links(month_url)
            except requests.HTTPError as e:
                print(f"Failed to list {month_url}: {e}")
                continue

            for file_url in files:
                filename = file_url.split('/')[-1]
                save_path = os.path.join(month_dir, filename)
                try:
                    download_file(file_url, save_path)
                except requests.HTTPError as e:
                    print(f"Failed to download {file_url}: {e}")

if __name__ == "__main__":
    main()
