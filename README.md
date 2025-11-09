# SHL Assessment Recommendation System

An intelligent hybrid RAG (Retrieval Augmented Generation) system for recommending SHL assessments based on natural language queries, job descriptions, or URLs. The system combines semantic search (Pinecone), graph database queries (Neo4j), and lexical matching (TF-IDF) to provide accurate and balanced assessment recommendations.

---

## 📊 Project Overview

This system takes a natural language query, job description text, or URL and returns a list of 5-10 most relevant "Individual Test Solutions" from SHL's assessment catalog. The system is evaluated on:
- **Recommendation Accuracy**: Mean Recall@10 against a provided test set
- **Recommendation Balance**: Balanced mix of assessments when queries span multiple domains (e.g., technical and behavioral skills)

---

## 🚀 Journey: From Baseline to Final Approach

### Phase 1: Initial Baseline (Recall@10: ~0.02)

**Approach:**
- Simple semantic search using Pinecone with OpenAI embeddings
- Basic query-to-assessment matching
- No entity extraction or query understanding

**Limitations:**
- Very low recall (0.02)
- No understanding of query intent
- No domain balancing
- Missing relevant assessments for multi-domain queries

---

### Phase 2: Entity Extraction & Few-Shot Learning (Recall@10: ~0.09-0.20)

**Improvements:**
- Added entity extraction from queries (role, skills, test types, duration, job levels)
- Implemented few-shot prompting using training data
- Cached training query embeddings for faster similarity matching
- Added keyword overlap boosting in descriptions
- Increased Pinecone `top_k` retrieval to 50

**Key Changes:**
- `extract_entities_from_query()` - Regex-based entity extraction
- `find_similar_queries()` - Semantic similarity for few-shot examples
- Description-based boosting for lexical matches

**Results:**
- Recall improved from 0.02 → 0.09-0.20
- Better handling of explicit skill mentions
- Still struggled with low-context queries (admin roles)

---

### Phase 3: Hybrid Retrieval with Neo4j (Recall@10: ~0.20-0.25)

**Improvements:**
- Integrated Neo4j graph database for structured lookups
- Graph-based retrieval by test types and job levels
- Related assessment discovery through graph relationships
- Dual Pinecone retrieval (technical + behavioral queries)
- Description embedding re-ranking with cosine similarity

**Key Changes:**
- Neo4j integration for structured metadata queries
- Separate semantic retrieval passes for tech and behavioral aspects
- Description-based re-ranking replacing keyword overlap
- Role expansion layer with rule-based keyword injection

**Results:**
- Recall improved to ~0.20-0.25
- Better multi-domain query handling
- Neo4j fallback for low-scoring Pinecone results

---

### Phase 4: Advanced Re-ranking & Normalization (Recall@10: ~0.25-0.35)

**Improvements:**
- URL path normalization for consistent matching
- TF-IDF + embedding fusion for lexical-semantic hybrid
- Exact-match rescue logic for training data matches
- Capped heuristic boosts to prevent score hijacking
- Neo4j fallback tuning (reduced base score, new test_type filtering)

**Key Changes:**
- `utils_normalize.py` - URL canonicalization
- `utils_similarity.py` - Consistent cosine similarity
- `fused_rerank.py` - TF-IDF + embedding fusion
- Exact-match rescue boosting (up to 0.45)
- Score clamping and normalization

**Results:**
- Recall improved to ~0.25-0.35
- More consistent URL matching
- Better lexical-semantic balance

---

### Phase 5: Final Approach - Four Key Improvements (Recall@10: ~0.38)

**Improvements:**
1. **Adaptive Fusion Weighting System**
2. **Intelligent Query Expansion**
3. **Explainability and Confidence Scoring**
4. **Enhanced Evaluation Infrastructure**

**Final Results:**
- **Mean Recall@10**: 0.377
- **Mean Precision@10**: 0.241
- **Mean MAP**: 0.299
- **Mean NDCG@10**: 0.446
- **Mean Domain Balance**: 0.917

---

## 🏗️ System Architecture

### Folder Structure

```
shl-assessment/
├── src/                          # Core retrieval logic and utilities
│   ├── hybrid_rag.py            # Main retrieval function
│   ├── query_classifier.py      # Query type classification
│   ├── query_expansion.py       # Role-based query expansion
│   ├── confidence_scorer.py     # Confidence scoring & explanations
│   ├── utils_normalize.py       # URL normalization
│   ├── utils_similarity.py      # Similarity functions
│   └── fused_rerank.py          # TF-IDF fusion
│
├── scripts/                      # Data loading and setup scripts
│   ├── upload_to_pinecone.py   # Upload to Pinecone
│   ├── load_data_to_neo.py     # Load to Neo4j
│   ├── ad_dd.py                # Scrape additional details
│   ├── shl_scrape.py           # Scrape catalog
│   └── maptt.py                # Map test types
│
├── evaluation/                    # Evaluation scripts
│   ├── evaluate_rag.py          # Basic evaluation
│   └── evaluate_rag_enhanced.py # Enhanced evaluation
│
└── data/                         # Data files
    ├── assessments_raw.json
    ├── traindata.csv
    └── ...
```

### Retrieval Pipeline

1. **Query Classification**: Classify query type (technical, behavioral, leadership, admin_clerical, multi_domain)
2. **Query Expansion**: Expand queries for low-performing roles with domain knowledge
3. **Entity Extraction**: Extract role, skills, test types, duration, job levels
4. **Dual Pinecone Retrieval**: Separate queries for technical and behavioral aspects
5. **TF-IDF Fusion**: Combine embedding and lexical scores
6. **Neo4j Enrichment**: Boost top results with graph relationships
7. **Entity-Based Boosting**: Boost matches for skills, test types, duration
8. **Exact-Match Rescue**: Strong boost for training data matches
9. **Diversity Re-ranking**: Ensure at least 3 unique test types in top results
10. **Confidence Scoring**: Generate confidence scores and explanations

---

## 🔧 Four Key Improvements

### 1. Adaptive Fusion Weighting System ✅

**Problem**: Static weights for all queries don't account for query-specific needs.

**Solution**: Rule-based query classifier that dynamically adjusts channel weights.

**Implementation:**
- **File**: `src/query_classifier.py`
- Classifies queries into: `technical`, `behavioral`, `leadership`, `admin_clerical`, `multi_domain`
- Uses keyword patterns and entity extraction

**Fusion Weight Configuration:**
```python
FUSION_WEIGHTS = {
    'technical': {'pinecone': 0.5, 'neo4j': 0.2, 'tfidf': 0.3},
    'behavioral': {'pinecone': 0.5, 'neo4j': 0.3, 'tfidf': 0.2},
    'leadership': {'pinecone': 0.4, 'neo4j': 0.5, 'tfidf': 0.1},
    'admin_clerical': {'pinecone': 0.4, 'neo4j': 0.3, 'tfidf': 0.3},
    'multi_domain': {'pinecone': 0.45, 'neo4j': 0.35, 'tfidf': 0.20}
}
```

**Impact**: Better recall for specialized query types (technical, leadership)

---

### 2. Intelligent Query Expansion ✅

**Problem**: Low-performing administrative and clerical role queries lack context.

**Solution**: Automatic query expansion with domain-specific skill clusters.

**Implementation:**
- **File**: `src/query_expansion.py`
- Domain knowledge base mapping roles to skill clusters
- Preserves original query intent while adding enrichment terms

**Role Expansion Examples:**
- **Admin Assistant** → adds: "office software proficiency, scheduling coordination, written communication, attention to detail, data entry accuracy"
- **Customer Service** → adds: "communication skills, conflict resolution, empathy, product knowledge, CRM systems"
- **Banking/Finance** → adds: "numerical accuracy, attention to detail, financial knowledge, regulatory compliance"

**Impact**: Improved recall for admin/clerical roles (target: +15-20%)

---

### 3. Explainability and Confidence Scoring ✅

**Problem**: No transparency in why assessments were recommended or confidence levels.

**Solution**: Multi-signal confidence scorer with human-readable explanations.

**Implementation:**
- **File**: `src/confidence_scorer.py`
- Confidence scoring (0.0-1.0) based on multiple signals
- Human-readable explanations referencing specific query aspects

**Confidence Signals:**
1. **Pinecone Score**: Semantic similarity (weight: 0.25)
2. **Neo4j Score**: Graph-based relationships (weight: 0.20)
3. **TF-IDF Score**: Lexical matches (weight: 0.20)
4. **Entity Match**: Query entity overlap (weight: 0.15)
5. **Query Overlap**: Keyword overlap (weight: 0.10)
6. **Domain Coverage**: Test type diversity (weight: 0.10)

**Explanation Format:**
- **High Confidence (≥0.8)**: "[High Confidence] Recommended because..."
- **Moderate Confidence (0.6-0.8)**: "[Moderate Confidence] Recommended because..."
- **Lower Confidence (<0.6)**: "[Lower Confidence] Recommended because..."

**Impact**: User trust through transparent explanations

---

### 4. Enhanced Evaluation Infrastructure ✅

**Problem**: Limited metrics (only Recall@10) and no query type breakdowns.

**Solution**: Comprehensive evaluation suite with multiple metrics and breakdowns.

**Implementation:**
- **File**: `evaluation/evaluate_rag_enhanced.py`
- Multiple metrics: Precision@10, MAP, NDCG@10, Domain Balance
- Query type breakdowns
- Timestamped result storage

**Metrics Calculated:**
1. **Recall@10**: Fraction of relevant items retrieved
2. **Precision@10**: Fraction of retrieved items that are relevant
3. **MAP (Mean Average Precision)**: Average precision across all relevant positions
4. **NDCG@10**: Normalized Discounted Cumulative Gain
5. **Domain Balance**: Measures technical/behavioral balance for multi-domain queries

**Query Type Breakdowns:**
- Metrics calculated separately for: `technical`, `behavioral`, `leadership`, `admin_clerical`, `multi_domain`
- Helps identify which query types need improvement

**Result Storage:**
- Saves to `evaluation_results_YYYYMMDD_HHMMSS.json`
- Includes summary statistics, per-query details, and query type breakdowns

**Impact**: Better tracking of system performance across query types

---

## 📈 Final Results

### Overall Performance
- **Mean Recall@10**: 0.377 (37.7% of relevant assessments retrieved)
- **Mean Precision@10**: 0.241 (24.1% of retrieved assessments are relevant)
- **Mean MAP**: 0.299
- **Mean NDCG@10**: 0.446
- **Mean Domain Balance**: 0.917 (excellent balance for multi-domain queries)

### Performance by Query Type

| Query Type | Count | Recall@10 | Precision@10 | MAP | NDCG@10 |
|------------|-------|-----------|--------------|-----|---------|
| **Technical** | 2 | 0.422 | 0.300 | 0.235 | 0.406 |
| **Behavioral** | 1 | 0.333 | 0.333 | 0.333 | 0.501 |
| **Leadership** | 1 | 0.800 | 0.400 | 0.800 | 0.869 |
| **Multi-Domain** | 5 | 0.324 | 0.191 | 0.244 | 0.395 |
| **Admin/Clerical** | 1 | 0.167 | 0.125 | 0.167 | 0.303 |

**Key Insights:**
- Leadership queries perform exceptionally well (0.80 recall)
- Technical queries show strong performance (0.42 recall)
- Multi-domain queries maintain good balance (0.92 domain balance)
- Admin/clerical queries need further improvement (0.17 recall)

---

## 🚀 Usage

### Setup

1. **Install Dependencies:**
```bash
pip install openai pinecone-client neo4j pandas numpy scikit-learn beautifulsoup4 requests python-dotenv tqdm
```

2. **Configure Environment Variables:**
Create a `.env` file with:
```
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=shl-assessment
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

3. **Load Data:**
```bash
# Scrape assessment catalog
python scripts/shl_scrape.py

# Map test types
python scripts/maptt.py

# Scrape additional details
python scripts/ad_dd.py

# Upload to Pinecone
python scripts/upload_to_pinecone.py

# Load to Neo4j
python scripts/load_data_to_neo.py
```

### Run Evaluation

**Standard Evaluation:**
```bash
cd evaluation
python evaluate_rag.py
```

**Enhanced Evaluation:**
```bash
cd evaluation
python evaluate_rag_enhanced.py
```

### Test Single Query

```python
import sys
sys.path.insert(0, 'src')
from hybrid_rag import hybrid_chat

results = hybrid_chat("I need a Java developer assessment", top_k=10, return_results=True)
for r in results:
    print(f"{r['name']}: Confidence={r['confidence']:.3f}")
    print(f"  Explanation: {r['explanation']}")
    print(f"  Link: {r['link']}")
```

---

## ⚙️ Configuration

### Adjust Fusion Weights
Edit `src/query_classifier.py` → `FUSION_WEIGHTS` dictionary

### Add Role Expansions
Edit `src/query_expansion.py` → `ROLE_EXPANSION_MAP` dictionary

### Tune Confidence Weights
Edit `src/confidence_scorer.py` → `weights` dictionary in `compute_confidence_score()`

### Modify Entity Extraction
Edit `src/hybrid_rag.py` → `extract_entities_from_query()` function

---

## 📁 Key Files

### Core Retrieval
- `src/hybrid_rag.py` - Main retrieval function with all improvements integrated

### Query Processing
- `src/query_classifier.py` - Query type classification and fusion weights
- `src/query_expansion.py` - Role-based query expansion
- `src/utils_normalize.py` - URL canonicalization
- `src/utils_similarity.py` - Cosine similarity functions

### Scoring & Ranking
- `src/confidence_scorer.py` - Confidence scoring and explanations
- `src/fused_rerank.py` - TF-IDF + embedding fusion

### Evaluation
- `evaluation/evaluate_rag.py` - Basic evaluation (Recall@10)
- `evaluation/evaluate_rag_enhanced.py` - Comprehensive evaluation suite

### Data Loading
- `scripts/upload_to_pinecone.py` - Upload assessments to Pinecone
- `scripts/load_data_to_neo.py` - Load assessments to Neo4j
- `scripts/ad_dd.py` - Scrape additional assessment details
- `scripts/shl_scrape.py` - Scrape assessment catalog
- `scripts/maptt.py` - Map test type codes to full names

---

## 🔄 Evolution Summary

| Phase | Approach | Recall@10 | Key Features |
|-------|----------|-----------|--------------|
| **1. Baseline** | Simple semantic search | ~0.02 | Pinecone only |
| **2. Entity Extraction** | Few-shot + entity extraction | ~0.09-0.20 | Entity extraction, few-shot learning |
| **3. Hybrid Retrieval** | Pinecone + Neo4j | ~0.20-0.25 | Graph database, dual retrieval |
| **4. Advanced Re-ranking** | Normalization + TF-IDF fusion | ~0.25-0.35 | URL normalization, TF-IDF fusion |
| **5. Final** | Four key improvements | **~0.38** | Adaptive fusion, query expansion, confidence scoring, enhanced metrics |

**Total Improvement**: 19x increase from baseline (0.02 → 0.38)

---

## 🚀 Deployment

### API Deployment

The API is ready for deployment to Render, Railway, Google Cloud Run, or AWS Lambda.

**Local Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables in .env file

# Run locally
cd api
uvicorn app:app --reload --port 8000
```

**Deploy to Render:**
1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `cd api && uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Add environment variables from `.env`

**API Endpoints:**
- `GET /health` - Health check endpoint
- `POST /recommend` - Get assessment recommendations

### Web Frontend

The web frontend is a single-page application. Update the `API_URL` in `web/index.html` with your deployed API URL, then deploy to Vercel, Netlify, or GitHub Pages.

### Generate Predictions CSV

```bash
python scripts/generate_predictions.py
```

Defaults to `data/testdata.xlsx` or specify with `--test-file` option.

---

## 📝 License

See LICENSE file for details.
