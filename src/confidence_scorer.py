def compute_confidence_score(result, query, entities):
    signals = {
        'pinecone_score': result.get('pinecone_score', result.get('embed_score', 0)),
        'neo4j_score': result.get('neo4j_score', result.get('neo_boost', 0)),
        'tfidf_score': result.get('tfidf_score', 0),
        'entity_match': 0.0,
        'query_overlap': 0.0,
        'domain_coverage': 0.0
    }
    
    max_pinecone = 1.0
    max_neo4j = 0.5
    max_tfidf = 1.0
    
    if max_pinecone > 0:
        signals['pinecone_score'] = min(signals['pinecone_score'] / max_pinecone, 1.0)
    if max_neo4j > 0:
        signals['neo4j_score'] = min(signals['neo4j_score'] / max_neo4j, 1.0)
    if max_tfidf > 0:
        signals['tfidf_score'] = min(signals['tfidf_score'] / max_tfidf, 1.0)
    
    name_lower = result.get('name', '').lower()
    desc_lower = result.get('description', '').lower()
    link_lower = result.get('link', '').lower()
    combined = f"{name_lower} {desc_lower} {link_lower}"
    
    entity_matches = 0
    total_entities = 0
    
    if entities.get('skills'):
        total_entities += len(entities['skills'])
        for skill in entities['skills']:
            if skill.lower() in combined:
                entity_matches += 1
    
    if entities.get('role'):
        total_entities += 1
        role_words = entities['role'].lower().split()
        if any(word in combined for word in role_words if len(word) > 3):
            entity_matches += 1
    
    if entities.get('test_types'):
        total_entities += len(entities['test_types'])
        result_test_types = result.get('test_types', [])
        if isinstance(result_test_types, str):
            result_test_types = [t.strip() for t in result_test_types.split(',')]
        for tt in entities['test_types']:
            if any(tt.lower() in str(rtt).lower() for rtt in result_test_types):
                entity_matches += 1
    
    if total_entities > 0:
        signals['entity_match'] = entity_matches / total_entities
    
    query_words = set(query.lower().split())
    result_text = f"{name_lower} {desc_lower}"
    result_words = set(result_text.split())
    overlap = len(query_words & result_words)
    signals['query_overlap'] = min(overlap / max(len(query_words), 1), 1.0)
    
    test_types = result.get('test_types', [])
    if isinstance(test_types, str):
        test_types = [t.strip() for t in test_types.split(',')]
    signals['domain_coverage'] = min(len(test_types) / 3.0, 1.0)
    
    weights = {
        'pinecone_score': 0.25,
        'neo4j_score': 0.20,
        'tfidf_score': 0.20,
        'entity_match': 0.15,
        'query_overlap': 0.10,
        'domain_coverage': 0.10
    }
    
    confidence = sum(signals[key] * weights[key] for key in signals)
    confidence = min(max(confidence, 0.0), 1.0)
    
    return confidence, signals


def generate_explanation(result, query, entities, confidence, signals):
    explanations = []
    
    if entities.get('role'):
        role_lower = entities['role'].lower()
        name_lower = result.get('name', '').lower()
        desc_lower = result.get('description', '').lower()
        if any(word in name_lower or word in desc_lower for word in role_lower.split() if len(word) > 3):
            explanations.append(f"matches the {entities['role']} role you specified")
    
    if entities.get('skills'):
        matched_skills = []
        name_lower = result.get('name', '').lower()
        desc_lower = result.get('description', '').lower()
        link_lower = result.get('link', '').lower()
        combined = f"{name_lower} {desc_lower} {link_lower}"
        
        for skill in entities['skills']:
            if skill.lower() in combined:
                matched_skills.append(skill)
        
        if matched_skills:
            if len(matched_skills) == 1:
                explanations.append(f"directly tests {matched_skills[0]} skills")
            else:
                explanations.append(f"covers {', '.join(matched_skills[:2])} and related skills")
    
    if entities.get('test_types'):
        result_test_types = result.get('test_types', [])
        if isinstance(result_test_types, str):
            result_test_types = [t.strip() for t in result_test_types.split(',')]
        
        matched_types = []
        for query_tt in entities['test_types']:
            for result_tt in result_test_types:
                if query_tt.lower() in str(result_tt).lower():
                    matched_types.append(str(result_tt))
                    break
        
        if matched_types:
            explanations.append(f"evaluates {matched_types[0]} competencies as requested")
    
    if signals.get('pinecone_score', 0) > 0.7:
        explanations.append("highly semantically similar to your requirements")
    
    if signals.get('neo4j_score', 0) > 0.3:
        explanations.append("strongly connected to your specified criteria in our knowledge graph")
    
    if signals.get('tfidf_score', 0) > 0.6:
        explanations.append("contains exact keyword matches with your query")
    
    if entities.get('duration') and result.get('duration'):
        try:
            result_dur = int(result.get('duration', 0))
            query_dur = int(entities['duration'])
            if abs(result_dur - query_dur) <= 10:
                explanations.append(f"matches your requested duration (~{result_dur} minutes)")
        except:
            pass
    
    if entities.get('job_levels') and result.get('job_levels'):
        result_levels = result.get('job_levels', [])
        if isinstance(result_levels, str):
            result_levels = [l.strip() for l in result_levels.split(',')]
        
        query_levels = entities['job_levels']
        if any(ql.lower() in ' '.join(str(rl) for rl in result_levels).lower() for ql in query_levels):
            explanations.append("appropriate for the experience level you specified")
    
    if not explanations:
        if confidence > 0.6:
            explanations.append("recommended based on overall relevance to your query")
        else:
            explanations.append("suggested as a potential match")
    
    if len(explanations) == 1:
        explanation = f"Recommended because this assessment {explanations[0]}."
    elif len(explanations) == 2:
        explanation = f"Recommended because this assessment {explanations[0]} and {explanations[1]}."
    else:
        explanation = f"Recommended because this assessment {explanations[0]}, {explanations[1]}, and {explanations[2]}."
    
    if confidence >= 0.8:
        explanation = f"[High Confidence] {explanation}"
    elif confidence >= 0.6:
        explanation = f"[Moderate Confidence] {explanation}"
    else:
        explanation = f"[Lower Confidence] {explanation}"
    
    return explanation


def add_confidence_and_explanations(results, query, entities):
    for result in results:
        confidence, signals = compute_confidence_score(result, query, entities)
        explanation = generate_explanation(result, query, entities, confidence, signals)
        
        result['confidence'] = confidence
        result['explanation'] = explanation
        result['confidence_signals'] = signals
    
    return results
