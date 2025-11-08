import os
import sys
import re
import pandas as pd
from collections import defaultdict
from functools import lru_cache
from openai import OpenAI
from neo4j import GraphDatabase
from pinecone import Pinecone
from dotenv import load_dotenv, dotenv_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_similarity import cosine_sim
from fused_rerank import build_tfidf_index, tfidf_scores, fuse_scores
from utils_normalize import canonical_path
from query_classifier import classify_query_type, get_fusion_weights
from query_expansion import expand_query_for_role
from confidence_scorer import add_confidence_and_explanations

env_vars = dotenv_values(".env")
load_dotenv(".env", override=True)

openai_api_key = env_vars.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
pinecone_api_key = env_vars.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
neo4j_uri = env_vars.get("NEO4J_URI") or os.getenv("NEO4J_URI")
neo4j_user = env_vars.get("NEO4J_USER") or os.getenv("NEO4J_USER")
neo4j_password = env_vars.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")
pinecone_index_name = env_vars.get("PINECONE_INDEX_NAME") or os.getenv("PINECONE_INDEX_NAME", "shl-assessment")

client = OpenAI(api_key=openai_api_key)
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
pinecone = Pinecone(api_key=pinecone_api_key)
index = pinecone.Index(pinecone_index_name)

training_data = None

def load_training_data():
    global training_data
    if training_data is not None:
        return training_data
    try:
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "traindata.csv")
        df = pd.read_csv(data_path)
        training_data = defaultdict(list)
        for _, row in df.iterrows():
            query = row["Query"]
            url = row["Assessment_url"]
            training_data[query].append(url)
        return training_data
    except:
        return {}

@lru_cache(maxsize=1)
def get_training_embeddings():
    training = load_training_data()
    if not training:
        return [], []
    queries = list(training.keys())
    embeddings = client.embeddings.create(model="text-embedding-3-small", input=queries).data
    vectors = [d.embedding for d in embeddings]
    return queries, vectors

def find_similar_queries(query, num_examples=3):
    queries, vectors = get_training_embeddings()
    if not queries:
        return []
    emb_data = client.embeddings.create(model="text-embedding-3-small", input=[query]).data
    query_emb = emb_data[0].embedding

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b)) / (
            (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
        )

    sims = [(queries[i], cosine(query_emb, vectors[i])) for i in range(len(queries))]
    sims.sort(key=lambda x: x[1], reverse=True)
    training = load_training_data()
    examples = []
    for train_query, _ in sims[:num_examples]:
        urls = training[train_query][:5]
        examples.append({"query": train_query, "assessments": urls})
    return examples


def extract_entities_from_query(text):
    t = text.strip()

    SKILLS = [
        "python", "java", "javascript", "typescript", "sql", "selenium", "html", "css",
        "react", "node", "jira", "confluence", "excel", "tableau", "power bi", "marketing",
        "salesforce", "seo", "writing", "communication", "leadership", "analytics",
        "testing", "qa", "manual", "automation", "data analysis", "deep learning",
        "machine learning", "tensorflow", "pytorch", "finance", "hr", "customer service",
        "content creation", "copywriting", "networking", "presentation", "negotiation"
    ]

    TEST_TYPE_KEYWORDS = [
        "cognitive", "personality", "behavioral", "aptitude", "leadership",
        "communication", "technical", "coding", "psychometric", "situational",
        "analytical", "verbal", "numerical", "reasoning"
    ]
    
    TEST_TYPE_MAPPING = {
        "cognitive": "Ability & Aptitude",
        "aptitude": "Ability & Aptitude",
        "analytical": "Ability & Aptitude",
        "verbal": "Ability & Aptitude",
        "numerical": "Ability & Aptitude",
        "reasoning": "Ability & Aptitude",
        "personality": "Personality & Behavior",
        "behavioral": "Personality & Behavior",
        "psychometric": "Personality & Behavior",
        "situational": "Biodata & Situational Judgement",
        "leadership": "Competencies",
        "communication": "Competencies",
        "technical": "Knowledge & Skills",
        "coding": "Knowledge & Skills"
    }

    DOMAINS = [
        "software", "marketing", "finance", "banking", "engineering", "data",
        "human resources", "operations", "it", "content", "sales", "customer service",
        "media", "design", "administration", "consulting", "education", "technology"
    ]
    
    JOB_LEVEL_MAPPING = {
        "entry": ["Entry-Level", "Graduate"],
        "junior": ["Entry-Level"],
        "mid": ["Mid-Professional", "Professional Individual Contributor"],
        "senior": ["Mid-Professional", "Professional Individual Contributor"],
        "lead": ["Manager", "Front Line Manager", "Supervisor"],
        "executive": ["Executive", "Director"],
        "manager": ["Manager", "Front Line Manager"],
        "director": ["Director", "Executive"],
        "graduate": ["Graduate", "Entry-Level"]
    }

    role = None
    role_match = re.search(
        r"(?:for|as|hire|hiring|require|looking\s+for|recruiting|need\s+(?:an?|the)?)\s+([A-Z][a-zA-Z/&\-\s]+?)(?=(?:,|\s+and|\n|with|who|that|at|in|$))",
        t,
        re.IGNORECASE,
    )
    if role_match:
        role = role_match.group(1).strip()
        role = re.sub(r"\b(and|with|the|a)\b.*$", "", role, flags=re.IGNORECASE).strip()
        words = role.split()
        if len(words) > 6:
            role = " ".join(words[:6])
        role = role.title()
    else:
        alt_role = re.search(
            r"(?<!^)(?:[A-Z][a-z]+(?:\s|/|&|-)){0,3}(Engineer|Manager|Developer|Analyst|Officer|Executive|Consultant|Writer|Lead|Administrator|Architect|Specialist|Head|Coordinator|Designer|Director|Supervisor|Recruiter|Strategist|Scientist|Associate)",
            t,
            re.IGNORECASE,
        )
        if alt_role:
            role = alt_role.group(0).strip().title()
            words = role.split()
            if len(words) > 6:
                role = " ".join(words[:6])

    duration = None
    dur_match = re.search(r"(\d+)\s*(?:mins?|minutes?|hours?|hrs?)", t, re.IGNORECASE)
    if dur_match:
        val = int(dur_match.group(1))
        duration = val * 60 if re.search(r"hour|hr", dur_match.group(0), re.IGNORECASE) else val

    exp_match = re.search(
        r"(entry|junior|mid|senior|lead|executive|manager|director|graduate)[ -]?(level)?",
        t,
        re.IGNORECASE,
    )
    job_levels = []
    if exp_match:
        exp_term = exp_match.group(1).lower()
        mapped_levels = JOB_LEVEL_MAPPING.get(exp_term, [])
        job_levels = mapped_levels.copy()

    skills = list(
        set(
            re.findall(
                r"\b(" + "|".join([re.escape(s) for s in SKILLS]) + r")\b", t, re.IGNORECASE
            )
        )
    )
    skills = [s.title() for s in skills]

    test_types_found = list(
        set(
            re.findall(
                r"\b(" + "|".join([re.escape(tf) for tf in TEST_TYPE_KEYWORDS]) + r")\b", t, re.IGNORECASE
            )
        )
    )
    test_types = []
    for keyword in test_types_found:
        mapped = TEST_TYPE_MAPPING.get(keyword.lower())
        if mapped and mapped not in test_types:
            test_types.append(mapped)

    domain = list(
        set(
            re.findall(
                r"\b(" + "|".join([re.escape(d) for d in DOMAINS]) + r")\b", t, re.IGNORECASE
            )
        )
    )
    domain = [d.title() for d in domain]

    lang_match = re.search(
        r"\b(english|spanish|french|hindi|german|mandarin|tamil|arabic|urdu|bengali)\b",
        t,
        re.IGNORECASE,
    )
    language = lang_match.group(1).capitalize() if lang_match else None

    return {
        "role": role if role else None,
        "skills": skills if skills else [],
        "duration": str(duration) if duration else None,
        "test_types": test_types if test_types else [],
        "job_levels": job_levels if job_levels else [],
        "domain": domain if domain else [],
        "language": language,
    }

def expand_query(query):
    query_lower = query.lower()
    expansions = []
    
    tech_map = {
        'java': ['java programming', 'java developer', 'java skills'],
        'python': ['python programming', 'python developer', 'python skills'],
        'sql': ['database', 'sql query', 'data management'],
        'developer': ['programming', 'coding', 'software development'],
        'analyst': ['data analysis', 'business analysis', 'analytical skills'],
        'sales': ['sales skills', 'sales representative', 'customer relations'],
        'manager': ['management', 'leadership', 'supervision'],
        'collaborate': ['teamwork', 'communication', 'interpersonal'],
        'graduate': ['entry level', 'junior', 'new hire'],
        'senior': ['experienced', 'expert', 'advanced']
    }
    
    for key, synonyms in tech_map.items():
        if key in query_lower:
            expansions.extend(synonyms)
    
    if expansions:
        expanded = f"{query} {' '.join(expansions[:3])}"
        return expanded
    return query

def hybrid_chat(query, top_k=10, return_results=False, quiet=False):
    entities = extract_entities_from_query(query)
    
    query_type = classify_query_type(query, entities)
    fusion_weights = get_fusion_weights(query_type)
    
    if not quiet:
        print(f"[Query Type: {query_type}] [Fusion Weights: Pinecone={fusion_weights['pinecone']:.2f}, "
              f"Neo4j={fusion_weights['neo4j']:.2f}, TF-IDF={fusion_weights['tfidf']:.2f}]")

    original_query = query
    query = expand_query_for_role(query, entities, expansion_weight=0.6)
    
    role_boost_map = {
        "manager": ["leadership", "competency", "supervision", "decision making"],
        "consultant": ["analytical thinking", "communication", "business acumen"],
        "engineer": ["technical", "automation", "problem solving", "software"],
        "developer": ["coding", "programming", "knowledge and skills"],
        "analyst": ["data interpretation", "aptitude", "reasoning", "problem solving"],
        "sales": ["communication", "negotiation", "interpersonal"],
        "marketing": ["creativity", "strategic thinking", "digital advertising"],
        "writer": ["verbal ability", "english comprehension", "content creation"]
    }

    for role, words in role_boost_map.items():
        if role in query.lower():
            query += " " + " ".join(words)

    tech_query = expand_query(query + " " + " ".join(entities.get("skills", [])))
    behavioral_query = query + " communication teamwork interpersonal leadership collaboration behavior personality"

    tech_emb = client.embeddings.create(model="text-embedding-3-small", input=tech_query).data[0].embedding
    beh_emb = client.embeddings.create(model="text-embedding-3-small", input=behavioral_query).data[0].embedding

    tech_results = index.query(vector=tech_emb, top_k=50, include_metadata=True)['matches']
    beh_results = index.query(vector=beh_emb, top_k=50, include_metadata=True)['matches']

    intent_weights = {"technical": 0.6, "behavioral": 0.4}
    merged_results = {}
    
    for match in tech_results:
        name = match['metadata'].get('name')
        if name:
            merged_results[name] = match['metadata'].copy()
            base_score = match['score'] * intent_weights['technical']
            merged_results[name]['score'] = base_score * fusion_weights['pinecone']
            merged_results[name]['embed_score'] = base_score * fusion_weights['pinecone']
            merged_results[name]['pinecone_score'] = match['score'] * fusion_weights['pinecone']
    
    for match in beh_results:
        name = match['metadata'].get('name')
        if name:
            if name in merged_results:
                base_score = match['score'] * intent_weights['behavioral']
                merged_results[name]['score'] += base_score * fusion_weights['pinecone']
                merged_results[name]['embed_score'] += base_score * fusion_weights['pinecone']
                merged_results[name]['pinecone_score'] += match['score'] * fusion_weights['pinecone']
            else:
                merged_results[name] = match['metadata'].copy()
                base_score = match['score'] * intent_weights['behavioral']
                merged_results[name]['score'] = base_score * fusion_weights['pinecone']
                merged_results[name]['embed_score'] = base_score * fusion_weights['pinecone']
                merged_results[name]['pinecone_score'] = match['score'] * fusion_weights['pinecone']

    all_results = list(merged_results.values())

    desc_to_result_idx = []
    descriptions = []
    for i, result in enumerate(all_results):
        desc = result.get('description', '')
        if desc and len(desc) > 20:
            descriptions.append(desc)
            desc_to_result_idx.append(i)
    
    if descriptions:
        vec, X = build_tfidf_index(descriptions)
        
        desc_embs = client.embeddings.create(model="text-embedding-3-small", input=descriptions).data
        embed_scores = []
        for desc_emb in desc_embs:
            sim = cosine_sim(tech_emb, desc_emb.embedding)
            embed_scores.append(sim)
        
        query_text = tech_query + " " + behavioral_query
        tfidf_sims = tfidf_scores(vec, X, query_text)
        
        w_embed = 1.0 - fusion_weights['tfidf']
        fused = fuse_scores(embed_scores, tfidf_sims, w_embed=w_embed, w_tfidf=fusion_weights['tfidf'])
        
        for idx, result_idx in enumerate(desc_to_result_idx):
            result = all_results[result_idx]
            fused_score = fused[idx] * fusion_weights['tfidf'] * 0.45
            result['score'] += fused_score
            result['desc_embed_score'] = embed_scores[idx]
            result['tfidf_score'] = tfidf_sims[idx] * fusion_weights['tfidf']

    top_names = [r.get('name') for r in all_results[:20] if r.get('name')]
    neo4j_added_count = 0
    existing_test_types = set()
    for r in all_results[:20]:
        tts = r.get('test_types', [])
        if isinstance(tts, str):
            tts = [t.strip() for t in tts.split(',')]
        existing_test_types.update(str(tt).lower() for tt in tts)
    
    with neo4j_driver.session() as session:
        if top_names:
            neo_data = session.run("""
                MATCH (a:Assessment)
                WHERE a.name IN $names
                OPTIONAL MATCH (a)-[:IS_OF_TYPE]->(tt:TestType)
                RETURN a.name AS name,
                       a.link AS link,
                       a.description AS description,
                       a.duration AS duration,
                       a.job_levels AS job_levels,
                       collect(tt.name) AS test_types
            """, names=top_names).data()

            neo_lookup = {r['name']: r for r in neo_data}
            for r in all_results:
                if r.get('name') in neo_lookup:
                    neo_info = neo_lookup[r['name']]
                    r['link'] = neo_info.get('link', r.get('link'))
                    r['description'] = neo_info.get('description', r.get('description'))
                    r['duration'] = neo_info.get('duration', r.get('duration'))
                    r['job_levels'] = neo_info.get('job_levels', r.get('job_levels'))
                    r['test_types'] = neo_info.get('test_types', r.get('test_types', []))
                    test_types = neo_info.get('test_types', [])
                    base_boost = 0.1 * len(test_types)
                    boost_val = base_boost * fusion_weights['neo4j']
                    r['score'] += boost_val
                    r['neo_boost'] = boost_val
                    r['neo4j_score'] = boost_val
        
        if entities.get('test_types') or entities.get('job_levels'):
            neo_fallback = []
            if entities.get('test_types'):
                for ttype in entities['test_types']:
                    data = session.run("""
                        MATCH (a:Assessment)-[:IS_OF_TYPE]->(t:TestType)
                        WHERE t.name = $ttype
                        OPTIONAL MATCH (a)-[:IS_OF_TYPE]->(tt:TestType)
                        RETURN a.name AS name, a.link AS link, a.description AS description,
                               a.duration AS duration, a.job_levels AS job_levels,
                               collect(tt.name) AS test_types
                        LIMIT 10
                    """, ttype=ttype).data()
                    neo_fallback.extend(data)
            
            for n in neo_fallback:
                name = n.get('name')
                if name and name not in merged_results:
                    new_tts = set(str(tt).lower() for tt in n.get('test_types', []))
                    if not existing_test_types.issuperset(new_tts):
                        base_score = 0.08
                        n['score'] = base_score * fusion_weights['neo4j']
                        n['neo_boost'] = base_score * fusion_weights['neo4j']
                        n['neo4j_score'] = base_score * fusion_weights['neo4j']
                        n['remote_support'] = 'No'
                        n['adaptive_support'] = 'No'
                        merged_results[name] = n
                        all_results.append(n)
                        neo4j_added_count += 1
                        existing_test_types.update(new_tts)

    if not quiet and neo4j_added_count > 0:
        print(f"  [Debug] Neo4j fallback added {neo4j_added_count} new assessments")

    for result in all_results:
        boost = 0.0
        name = result.get('name', '').lower()
        desc = result.get('description', '').lower()
        link = result.get('link', '').lower()

        for s in entities.get('skills', []):
            if s.lower() in name or s.lower() in desc or s.lower() in link:
                boost += 0.15

        for d in entities.get('domain', []):
            if d.lower() in name or d.lower() in desc:
                boost += 0.1

        result_test_types = result.get('test_types', [])
        if isinstance(result_test_types, str):
            result_test_types = [t.strip() for t in result_test_types.split(',')]
        for tf in entities.get('test_types', []):
            if tf.lower() in ' '.join(str(tt) for tt in result_test_types).lower():
                boost += 0.3

        if entities.get('duration') and result.get('duration'):
            try:
                if abs(int(result.get('duration')) - int(entities['duration'])) <= 10:
                    boost += 0.15
            except:
                pass

        boost = min(boost, 0.4)
        result['score'] += boost
        result['boost'] = boost

    MAX_ADDITIONAL_BOOST = 0.45
    training = load_training_data()
    expected_training_paths = set()
    if query in training:
        for url in training[query]:
            path = canonical_path(url)
            if path:
                expected_training_paths.add(path)
    else:
        similar_queries = find_similar_queries(query, num_examples=3)
        for ex in similar_queries:
            train_query = ex.get('query')
            if train_query in training:
                for url in training[train_query]:
                    path = canonical_path(url)
                    if path:
                        expected_training_paths.add(path)
    
    for r in all_results:
        item_path = canonical_path(r.get('link') or r.get('url', ''))
        rescue_boost = 0.0
        for expected in expected_training_paths:
            if expected and item_path:
                if expected == item_path or expected in item_path or item_path in expected:
                    rescue_boost += 0.6
                    break
        if rescue_boost > 0:
            r['score'] += min(rescue_boost, MAX_ADDITIONAL_BOOST)
            r['rescue_boost'] = min(rescue_boost, MAX_ADDITIONAL_BOOST)

    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    final_results = []
    seen_types = set()
    for res in all_results:
        tts = tuple(res.get('test_types', []))
        if len(seen_types) < 3 or not seen_types.issuperset(tts):
            final_results.append(res)
            seen_types.update(tts)
        if len(final_results) >= top_k:
            break

    final_results = add_confidence_and_explanations(final_results, original_query, entities)

    if not quiet:
        print("\n" + "="*70)
        print("TOP 5 RECOMMENDATIONS:")
        print("="*70)
        for i, r in enumerate(final_results[:5], 1):
            embed_s = r.get('embed_score', 0)
            tfidf_s = r.get('tfidf_score', 0)
            boost_s = r.get('boost', 0)
            neo_s = r.get('neo_boost', 0)
            rescue_s = r.get('rescue_boost', 0)
            confidence = r.get('confidence', 0)
            total = r.get('score', 0)
            print(f"\n{i}. {r.get('name', 'N/A')}")
            print(f"   Total: {total:.3f} | Confidence: {confidence:.3f} | Embed: {embed_s:.3f} | "
                  f"TFIDF: {tfidf_s:.3f} | Boost: {boost_s:.3f} | Neo4j: {neo_s:.3f} | Rescue: {rescue_s:.3f}")
            print(f"   Link: {r.get('link', 'N/A')}")
            print(f"   Test Types: {', '.join(r.get('test_types', [])) if r.get('test_types') else 'N/A'}")
            if r.get('explanation'):
                print(f"   Explanation: {r.get('explanation')}")

    if return_results:
        return final_results[:top_k]

    context_text = "Available assessments:\n"
    for i, res in enumerate(final_results[:10], 1):
        context_text += f"{i}. {res.get('name', 'N/A')} - {res.get('link', 'N/A')}\n"
        if res.get('description'):
            context_text += f"   Description: {res.get('description', '')[:120]}...\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful Talent Acquisition Specialist that recommends SHL assessments based on job requirements. Use the context below to recommend the best assessments."},
            {"role": "user", "content": f"Query: {query}\n\n{context_text}\n\nRecommend the most relevant assessments with reasons."}
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        try:
            answer = hybrid_chat(query)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
    
    neo4j_driver.close()
