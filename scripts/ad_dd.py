import json
import os
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time

def scrape_assessment_details(url, print_html=False):
    try:
        resp = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        if print_html:
            print("="*80)
            print(f"RAW HTML FOR URL: {url}")
            print("="*80)
            print(resp.text)
            print("="*80)
            return "", "", []
        
        description = ""
        duration = ""
        job_levels = []
        
        rows = soup.find_all('div', class_='product-catalogue-training-calendar__row')
        
        for row in rows:
            h4 = row.find('h4')
            if h4 and h4.get_text(strip=True).lower() == 'description':
                p = row.find('p')
                if p:
                    description = p.get_text(strip=True)
                    break
        
        for row in rows:
            h4 = row.find('h4')
            if h4 and 'job level' in h4.get_text(strip=True).lower():
                p = row.find('p')
                if p:
                    levels_text = p.get_text(strip=True)
                    job_levels = [l.strip() for l in levels_text.split(',') if l.strip()]
                    break
        
        for row in rows:
            h4 = row.find('h4')
            if h4 and 'assessment length' in h4.get_text(strip=True).lower():
                p = row.find('p')
                if p:
                    duration_text = p.get_text(strip=True)
                    match = re.search(r'=\s*(\d+)', duration_text)
                    if match:
                        duration = match.group(1)
                    else:
                        match = re.search(r'(\d+)', duration_text)
                        if match:
                            duration = match.group(1)
                        else:
                            duration = duration_text
                    break
        
        return description[:1000] if description else "", duration if duration else "", job_levels
    
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "", "", []

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "assessments_raw.json")
    output_file = os.path.join(base_dir, "data", "assessments_raw.json")
    
    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        assessments = json.load(f)
    
    print(f"Found {len(assessments)} assessments")
    
    for assessment in tqdm(assessments, desc="Scraping"):
        url = assessment.get("url", "")
        if not url:
            assessment["description"] = ""
            assessment["duration"] = ""
            assessment["job_levels"] = []
            continue
        
        if assessment.get("description") and assessment.get("duration") and assessment.get("job_levels"):
            continue
        
        description, duration, job_levels = scrape_assessment_details(url)
        assessment["description"] = description
        assessment["duration"] = duration
        assessment["job_levels"] = job_levels
        
        time.sleep(0.5)
    
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
