import requests
import json
import time

def fetch_json_data(base_url, num_pages=10):
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0',
        'Accept': 'application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1',
    })

    # Warm up the session by visiting the site homepage first
    try:
        session.get('https://streetcarsuburbs.news/', timeout=15)
        time.sleep(5)
    except Exception:
        pass

    all_data = []  # List to hold all data collected

    # Loop over the specified number of pages
    for page in range(1, num_pages + 1):
        time.sleep(10)
        # Request 100 items per page to reduce total number of requests
        page_url = f"{base_url}?page={page}&per_page=100"
        print(page_url)

        # Retry loop with exponential backoff on 429
        max_retries = 5
        for attempt in range(max_retries):
            response = session.get(page_url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                all_data.extend(data)
                print(f"Data retrieved from page {page}")
                break
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60 * (2 ** attempt)))
                print(f"429 on page {page}, attempt {attempt + 1}/{max_retries}. Waiting {retry_after}s...")
                time.sleep(retry_after)
            else:
                print(f"Failed to retrieve data from page {page}: Status Code {response.status_code}")
                break
        else:
            print(f"Giving up on page {page} after {max_retries} attempts.")

    return all_data

# URL to the JSON data
base_url = 'https://streetcarsuburbs.news/wp-json/wp/v2/posts'
# Fetch JSON data from the first 100 pages
data = fetch_json_data(base_url, num_pages=10)

# Optionally, save the data to a file or handle it as needed
with open('streetcarsuburbs.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Data retrieval complete. JSON file saved.")
