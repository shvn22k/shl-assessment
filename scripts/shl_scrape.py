import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import time
import json

link_og = "https://www.shl.com/products/product-catalog/?start=0&type=1&type=1"

def scrape_catalog(url):
    resp = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    table = None
    for candidate in soup.find_all("table"):
        heading_cell = candidate.find("th", string=lambda s: s and "Individual Test Solutions" in s)
        if heading_cell:
            table = candidate
            break

    if not table:
        raise ValueError("Could not locate the 'Individual Test Solutions' table on the page.")
    rows = table.find_all('tr')
    items = []
    headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) != len(headers):
            continue
        name = cols[0].get_text(strip=True)
        
        remote_tag = cols[1] if len(cols) > 1 else None
        adaptive_tag = cols[2] if len(cols) > 2 else None
        
        remote_span = remote_tag.find("span", class_="catalogue__circle") if remote_tag else None
        adaptive_span = adaptive_tag.find("span", class_="catalogue__circle") if adaptive_tag else None
        
        remote_yes = bool(remote_span and "-yes" in remote_span.get("class", []))
        adaptive_yes = bool(adaptive_span and "-yes" in adaptive_span.get("class", []))
        
        test_type = cols[3].get_text(strip=True) if len(cols) > 3 else ""
        
        link_tag = cols[0].find("a")
        link = urljoin(url, link_tag["href"]) if link_tag and link_tag.get("href") else ""
        
        item = {
            "url": link,
            "name": name,
            "adaptive_support": "Yes" if adaptive_yes else "No",
            "remote_support": "Yes" if remote_yes else "No",
            "test_types": test_type
        }
        items.append(item)
    return items

def catalog_to_json(items, *, filepath="data/assessments_raw.json", indent=2, ensure_ascii=False):
    payload = json.dumps(items, ensure_ascii=ensure_ascii, indent=indent)
    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(payload)
    return payload

if __name__ == "__main__":
    items_list = []
    for i in range(0,373, 12):
        items = scrape_catalog(f"https://www.shl.com/products/product-catalog/?start={i}&type=1&type=1")
        print(f"Found {len(items)} items - {i}")
        items_list.extend(items)
    json_payload = catalog_to_json(items_list)
    print(json_payload)
