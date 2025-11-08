import json
import os
import re
import time
import unicodedata
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, dotenv_values

env_vars = dotenv_values(".env")
load_dotenv(".env", override=True)

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "assessments_raw.json")
BATCH_SIZE = 30
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "shl-assessment")
VECTOR_DIM = int(os.getenv("PINECONE_VECTOR_DIM", "1536"))
EMBED_MODEL = "text-embedding-3-small"
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

openai_api_key = env_vars.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
pinecone_api_key = env_vars.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found")

client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(INDEX_NAME)

def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def create_ascii_id(name):
    name = unicodedata.normalize('NFKD', name)
    replacements = {
        '–': '-',
        '—': '-',
        '…': '...',
        '°': 'deg',
        '®': '',
        '©': '',
        '™': '',
    }
    for unicode_char, ascii_char in replacements.items():
        name = name.replace(unicode_char, ascii_char)
    name = name.encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name or "unknown"

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    items = []
    for a in assessments:
        test_types = a.get("test_types", [])
        if isinstance(test_types, list):
            test_type_str = ", ".join(test_types)
        else:
            test_type_str = str(test_types) if test_types else "N/A"
        
        job_levels = a.get("job_levels", [])
        if isinstance(job_levels, list):
            job_levels_str = ", ".join(job_levels)
        else:
            job_levels_str = str(job_levels) if job_levels else "N/A"
        
        description = a.get("description", "")
        duration = a.get("duration", "")
        
        semantic_text = f"""
Assessment Name: {a.get('name', 'N/A')}
Description: {description}
Test Types: {test_type_str}
Job Levels: {job_levels_str}
Duration: {duration} minutes
Remote Testing: {a.get('remote_support', 'No')}
Adaptive/IRT: {a.get('adaptive_support', 'No')}
Link: {a.get('url', 'N/A')}
"""
        
        meta = {
            "name": a.get("name", "N/A"),
            "description": description,
            "test_types": test_type_str,
            "job_levels": job_levels_str,
            "duration": duration,
            "remote_support": a.get("remote_support", "No"),
            "adaptive_support": a.get("adaptive_support", "No"),
            "link": a.get("url", "N/A")
        }
        
        item_id = create_ascii_id(a.get("name", "unknown"))
        items.append((item_id, semantic_text.strip(), meta))

    print(f"Uploading {len(items)} assessments...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = embed_texts(texts)
        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors=vectors)
        time.sleep(0.3)

    print("Done!")

if __name__ == "__main__":
    main()
