import os
import sys
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from src.hybrid_rag import hybrid_chat
from src.utils_normalize import canonical_path

def extract_url_path(url):
    if not url:
        return None
    return canonical_path(url)

def load_test_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "traindata.csv")
    df = pd.read_csv(filepath)
    query_to_urls = defaultdict(list)
    for _, row in df.iterrows():
        query = row['Query']
        url = row['Assessment_url']
        query_to_urls[query].append(extract_url_path(url))
    return query_to_urls

def calculate_recall_at_k(recommended_urls, expected_urls, k=10):
    if not expected_urls:
        return 0.0
    recommended_set = set(recommended_urls[:k])
    expected_set = set(expected_urls)
    intersection = recommended_set & expected_set
    recall = len(intersection) / len(expected_set) if expected_set else 0.0
    return recall

def evaluate():
    print("Loading test data...")
    query_to_urls = load_test_data()
    print(f"Evaluating on {len(query_to_urls)} unique queries...")
    print("="*70)
    
    recalls = []
    total_queries = len(query_to_urls)
    
    for i, (query, expected_urls) in enumerate(query_to_urls.items(), 1):
        print(f"\n[{i}/{total_queries}] Query: {query[:80]}...")
        
        try:
            results = hybrid_chat(query, top_k=10, return_results=True, quiet=True)
            
            recommended_urls = []
            for result in results:
                link = result.get('link', '')
                if link:
                    recommended_urls.append(extract_url_path(link))
            
            recommended_set = set(recommended_urls[:10])
            expected_set = set(expected_urls)
            intersection = recommended_set & expected_set
            
            recall = len(intersection) / len(expected_set) if expected_set else 0.0
            recalls.append(recall)
            
            print(f"\n  Expected: {len(expected_urls)}")
            for url in expected_urls:
                print(f"    - {url}")
            
            print(f"\n  Recommended: {len(recommended_urls)}")
            for url in recommended_urls:
                print(f"    - {url}")
            
            print(f"\n  Found: {len(intersection)}")
            if intersection:
                for url in intersection:
                    print(f"    - {url}")
            else:
                print("    (none)")
            
            print(f"\n  Recall@10: {recall:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            recalls.append(0.0)
    
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Total Queries: {total_queries}")
    print(f"Mean Recall@10: {mean_recall:.4f}")
    print(f"Min Recall@10: {min(recalls):.4f}")
    print(f"Max Recall@10: {max(recalls):.4f}")
    print("="*70)
    
    return mean_recall

if __name__ == "__main__":
    evaluate()
