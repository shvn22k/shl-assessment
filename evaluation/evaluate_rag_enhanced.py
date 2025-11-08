import os
import sys
import pandas as pd
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from src.hybrid_rag import hybrid_chat
from src.utils_normalize import canonical_path
from src.query_classifier import classify_query_type

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

def calculate_precision_at_k(recommended_urls, expected_urls, k=10):
    if not recommended_urls:
        return 0.0
    recommended_set = set(recommended_urls[:k])
    expected_set = set(expected_urls)
    intersection = recommended_set & expected_set
    precision = len(intersection) / len(recommended_set) if recommended_set else 0.0
    return precision

def calculate_recall_at_k(recommended_urls, expected_urls, k=10):
    if not expected_urls:
        return 0.0
    recommended_set = set(recommended_urls[:k])
    expected_set = set(expected_urls)
    intersection = recommended_set & expected_set
    recall = len(intersection) / len(expected_set) if expected_set else 0.0
    return recall

def calculate_map(recommended_urls, expected_urls, k=10):
    if not expected_urls:
        return 0.0
    expected_set = set(expected_urls)
    relevant_positions = []
    for i, url in enumerate(recommended_urls[:k], 1):
        if url in expected_set:
            relevant_positions.append(i)
    if not relevant_positions:
        return 0.0
    precisions = []
    for i, pos in enumerate(relevant_positions, 1):
        precision = i / pos
        precisions.append(precision)
    return sum(precisions) / len(expected_set) if expected_set else 0.0

def calculate_ndcg(recommended_urls, expected_urls, k=10):
    if not expected_urls:
        return 0.0
    expected_set = set(expected_urls)
    dcg = 0.0
    for i, url in enumerate(recommended_urls[:k], 1):
        if url in expected_set:
            relevance = 1.0
            dcg += relevance / (np.log2(i + 1))
    idcg = 0.0
    num_relevant = min(len(expected_set), k)
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / (np.log2(i + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg

def calculate_domain_balance(results, query_type):
    if query_type != 'multi_domain':
        return 1.0
    technical_types = {'Knowledge & Skills', 'Ability & Aptitude'}
    behavioral_types = {'Personality & Behavior', 'Competencies', 'Development & 360'}
    tech_count = 0
    beh_count = 0
    for result in results:
        test_types = result.get('test_types', [])
        if isinstance(test_types, str):
            test_types = [t.strip() for t in test_types.split(',')]
        for tt in test_types:
            if any(tech in str(tt) for tech in technical_types):
                tech_count += 1
            if any(beh in str(tt) for beh in behavioral_types):
                beh_count += 1
    total = tech_count + beh_count
    if total == 0:
        return 0.0
    tech_ratio = tech_count / total
    balance_score = 1.0 - abs(tech_ratio - 0.5) * 2
    return max(balance_score, 0.0)

def evaluate_enhanced():
    print("Loading test data...")
    query_to_urls = load_test_data()
    print(f"Evaluating on {len(query_to_urls)} unique queries...")
    print("="*70)
    
    metrics = {
        'recalls': [],
        'precisions': [],
        'maps': [],
        'ndcgs': [],
        'domain_balances': [],
        'by_query_type': defaultdict(lambda: {
            'recalls': [],
            'precisions': [],
            'maps': [],
            'ndcgs': []
        })
    }
    
    query_details = []
    total_queries = len(query_to_urls)
    
    for i, (query, expected_urls) in enumerate(query_to_urls.items(), 1):
        print(f"\n[{i}/{total_queries}] Query: {query[:80]}...")
        
        try:
            from src.hybrid_rag import extract_entities_from_query
            entities = extract_entities_from_query(query)
            query_type = classify_query_type(query, entities)
            
            results = hybrid_chat(query, top_k=10, return_results=True, quiet=True)
            
            recommended_urls = []
            for result in results:
                link = result.get('link', '')
                if link:
                    recommended_urls.append(extract_url_path(link))
            
            recall = calculate_recall_at_k(recommended_urls, expected_urls, k=10)
            precision = calculate_precision_at_k(recommended_urls, expected_urls, k=10)
            map_score = calculate_map(recommended_urls, expected_urls, k=10)
            ndcg = calculate_ndcg(recommended_urls, expected_urls, k=10)
            domain_balance = calculate_domain_balance(results, query_type)
            
            metrics['recalls'].append(recall)
            metrics['precisions'].append(precision)
            metrics['maps'].append(map_score)
            metrics['ndcgs'].append(ndcg)
            metrics['domain_balances'].append(domain_balance)
            
            metrics['by_query_type'][query_type]['recalls'].append(recall)
            metrics['by_query_type'][query_type]['precisions'].append(precision)
            metrics['by_query_type'][query_type]['maps'].append(map_score)
            metrics['by_query_type'][query_type]['ndcgs'].append(ndcg)
            
            query_details.append({
                'query': query,
                'query_type': query_type,
                'recall': recall,
                'precision': precision,
                'map': map_score,
                'ndcg': ndcg,
                'domain_balance': domain_balance,
                'expected_count': len(expected_urls),
                'recommended_count': len(recommended_urls),
                'found_count': len(set(recommended_urls[:10]) & set(expected_urls))
            })
            
            print(f"  Query Type: {query_type}")
            print(f"  Recall@10: {recall:.4f} | Precision@10: {precision:.4f} | "
                  f"MAP: {map_score:.4f} | NDCG@10: {ndcg:.4f} | Domain Balance: {domain_balance:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            metrics['recalls'].append(0.0)
            metrics['precisions'].append(0.0)
            metrics['maps'].append(0.0)
            metrics['ndcgs'].append(0.0)
            metrics['domain_balances'].append(0.0)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Total Queries: {total_queries}")
    print(f"\nOverall Metrics:")
    print(f"  Mean Recall@10:    {np.mean(metrics['recalls']):.4f}")
    print(f"  Mean Precision@10:  {np.mean(metrics['precisions']):.4f}")
    print(f"  Mean MAP:           {np.mean(metrics['maps']):.4f}")
    print(f"  Mean NDCG@10:       {np.mean(metrics['ndcgs']):.4f}")
    print(f"  Mean Domain Balance: {np.mean(metrics['domain_balances']):.4f}")
    print(f"\nMin/Max:")
    print(f"  Min Recall@10:     {np.min(metrics['recalls']):.4f}")
    print(f"  Max Recall@10:     {np.max(metrics['recalls']):.4f}")
    
    print(f"\nBreakdown by Query Type:")
    for qtype, type_metrics in metrics['by_query_type'].items():
        if type_metrics['recalls']:
            print(f"\n  {qtype.upper()}:")
            print(f"    Count: {len(type_metrics['recalls'])}")
            print(f"    Mean Recall@10:    {np.mean(type_metrics['recalls']):.4f}")
            print(f"    Mean Precision@10:  {np.mean(type_metrics['precisions']):.4f}")
            print(f"    Mean MAP:           {np.mean(type_metrics['maps']):.4f}")
            print(f"    Mean NDCG@10:       {np.mean(type_metrics['ndcgs']):.4f}")
    
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{timestamp}.json"
    
    results_data = {
        'timestamp': timestamp,
        'summary': {
            'total_queries': total_queries,
            'mean_recall': float(np.mean(metrics['recalls'])),
            'mean_precision': float(np.mean(metrics['precisions'])),
            'mean_map': float(np.mean(metrics['maps'])),
            'mean_ndcg': float(np.mean(metrics['ndcgs'])),
            'mean_domain_balance': float(np.mean(metrics['domain_balances']))
        },
        'by_query_type': {
            qtype: {
                'count': len(type_metrics['recalls']),
                'mean_recall': float(np.mean(type_metrics['recalls'])),
                'mean_precision': float(np.mean(type_metrics['precisions'])),
                'mean_map': float(np.mean(type_metrics['maps'])),
                'mean_ndcg': float(np.mean(type_metrics['ndcgs']))
            }
            for qtype, type_metrics in metrics['by_query_type'].items()
            if type_metrics['recalls']
        },
        'query_details': query_details
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results_data

if __name__ == "__main__":
    evaluate_enhanced()
