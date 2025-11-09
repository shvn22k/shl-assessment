import re

# keyword lists for classification
TECHNICAL_KEYWORDS = [
    "java", "python", "sql", "javascript", "coding", "programming", "developer",
    "engineer", "technical", "software", "automation", "selenium", "testing",
    "data", "analyst", "database", "api", "framework", "algorithm"
]

BEHAVIORAL_KEYWORDS = [
    "communication", "teamwork", "collaboration", "interpersonal", "personality",
    "behavioral", "emotional", "empathy", "conflict", "customer service",
    "sales", "negotiation", "persuasion"
]

LEADERSHIP_KEYWORDS = [
    "leadership", "manager", "director", "executive", "coo", "ceo", "supervisor",
    "management", "strategic", "decision making", "team management", "supervision",
    "lead", "head", "chief", "vp", "vice president"
]

ADMIN_CLERICAL_KEYWORDS = [
    "admin", "administrative", "assistant", "clerical", "secretary", "receptionist",
    "data entry", "office", "scheduling", "coordination", "filing", "bookkeeping",
    "administrator", "clerk", "support staff"
]


def classify_query_type(query, entities):
    query_lower = query.lower()
    
    # count keywords
    tech_count = sum(1 for kw in TECHNICAL_KEYWORDS if kw in query_lower)
    behavioral_count = sum(1 for kw in BEHAVIORAL_KEYWORDS if kw in query_lower)
    leadership_count = sum(1 for kw in LEADERSHIP_KEYWORDS if kw in query_lower)
    admin_count = sum(1 for kw in ADMIN_CLERICAL_KEYWORDS if kw in query_lower)
    
    # boost from role
    if entities.get('role'):
        role_lower = entities['role'].lower()
        if any(kw in role_lower for kw in TECHNICAL_KEYWORDS):
            tech_count += 2
        if any(kw in role_lower for kw in LEADERSHIP_KEYWORDS):
            leadership_count += 2
        if any(kw in role_lower for kw in ADMIN_CLERICAL_KEYWORDS):
            admin_count += 2
    
    # boost from skills
    if entities.get('skills'):
        for skill in entities['skills']:
            skill_lower = skill.lower()
            if any(kw in skill_lower for kw in TECHNICAL_KEYWORDS):
                tech_count += 1
    
    counts = {
        'technical': tech_count,
        'behavioral': behavioral_count,
        'leadership': leadership_count,
        'admin_clerical': admin_count
    }
    
    max_count = max(counts.values())
    if max_count == 0:
        return 'multi_domain'
    
    # check if multi-domain
    non_zero = [k for k, v in counts.items() if v > 0]
    if len(non_zero) >= 2 and max_count >= 2:
        return 'multi_domain'
    
    # return highest
    for qtype, count in counts.items():
        if count == max_count:
            return qtype
    
    return 'multi_domain'


FUSION_WEIGHTS = {
    'technical': {'pinecone': 0.5, 'neo4j': 0.2, 'tfidf': 0.3},
    'behavioral': {'pinecone': 0.5, 'neo4j': 0.3, 'tfidf': 0.2},
    'leadership': {'pinecone': 0.4, 'neo4j': 0.5, 'tfidf': 0.1},
    'admin_clerical': {'pinecone': 0.4, 'neo4j': 0.3, 'tfidf': 0.3},
    'multi_domain': {'pinecone': 0.45, 'neo4j': 0.35, 'tfidf': 0.20}
}


def get_fusion_weights(query_type):
    return FUSION_WEIGHTS.get(query_type, FUSION_WEIGHTS['multi_domain'])
