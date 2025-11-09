import os
import sys
import pandas as pd

# add paths
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)
sys.path.insert(0, os.path.join(base, 'src'))

from hybrid_rag import hybrid_chat

def load_test_queries(filepath):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        if 'Query' in df.columns:
            return df['Query'].unique().tolist()
        else:
            # first column
            return df.iloc[:, 0].unique().tolist()
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
        if 'Query' in df.columns:
            return df['Query'].unique().tolist()
        else:
            # first column
            return df.iloc[:, 0].unique().tolist()
    elif filepath.endswith('.json'):
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                queries = []
                for item in data:
                    q = item.get('query') or item.get('Query', '')
                    if q:
                        queries.append(q)
                return queries
            elif isinstance(data, dict) and 'queries' in data:
                return data['queries']
    return []

def generate_predictions(test_queries_file, output_file='predictions.csv'):
    print(f"Loading test queries from {test_queries_file}...")
    
    if not os.path.exists(test_queries_file):
        print(f"Error: Test queries file not found: {test_queries_file}")
        print("Please provide the path to your test queries file (CSV, XLSX, or JSON)")
        return
    
    queries = load_test_queries(test_queries_file)
    
    if not queries:
        print("Error: No queries found in the test file")
        return
    
    print(f"Found {len(queries)} unique queries")
    print("Generating recommendations...")
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Processing: {query[:80]}...")
        
        try:
            recs = hybrid_chat(query, top_k=10, return_results=True, quiet=True)
            
            for rec in recs:
                url = rec.get('link') or rec.get('url', '')
                if url:
                    results.append({
                        'Query': query,
                        'Assessment_url': url
                    })
        
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not results:
        print("Error: No recommendations generated")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nPredictions saved to {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Unique queries: {df['Query'].nunique()}")
    print(f"Average recommendations per query: {len(df) / df['Query'].nunique():.2f}")
    
    print("\nFirst few rows:")
    print(df.head(10).to_string())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate predictions CSV for test queries')
    parser.add_argument('--test-file', type=str, help='Path to test queries file (CSV, XLSX, or JSON). Default: data/testdata.xlsx')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # default to testdata.xlsx if not provided
    if not args.test_file:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_file = os.path.join(base, 'data', 'testdata.xlsx')
        if os.path.exists(test_file):
            args.test_file = test_file
        else:
            print("Error: --test-file is required or testdata.xlsx must exist in data/ folder")
            print("Usage: python generate_predictions.py --test-file <path_to_test_file>")
            sys.exit(1)
    
    generate_predictions(args.test_file, args.output)

