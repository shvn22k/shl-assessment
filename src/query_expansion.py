ROLE_EXPANSION_MAP = {
    'admin': [
        'office software proficiency', 'scheduling coordination', 
        'written communication', 'attention to detail', 
        'data entry accuracy', 'administrative support', 'filing systems'
    ],
    'administrative': [
        'office software proficiency', 'scheduling coordination',
        'written communication', 'attention to detail',
        'data entry accuracy', 'administrative support', 'filing systems'
    ],
    'assistant': [
        'office software proficiency', 'scheduling coordination',
        'written communication', 'attention to detail',
        'administrative support', 'multitasking'
    ],
    'administrator': [
        'office software proficiency', 'scheduling coordination',
        'written communication', 'attention to detail',
        'administrative support', 'organizational skills'
    ],
    'secretary': [
        'office software proficiency', 'scheduling coordination',
        'written communication', 'attention to detail',
        'administrative support', 'filing systems', 'multitasking'
    ],
    'receptionist': [
        'communication skills', 'customer service', 'scheduling coordination',
        'multitasking', 'phone etiquette', 'interpersonal skills'
    ],
    'clerk': [
        'data entry accuracy', 'attention to detail', 'filing systems',
        'office software proficiency', 'organizational skills'
    ],
    'customer service': [
        'communication skills', 'conflict resolution', 'empathy',
        'product knowledge', 'crm systems', 'problem solving',
        'patience', 'active listening'
    ],
    'customer support': [
        'communication skills', 'conflict resolution', 'empathy',
        'product knowledge', 'crm systems', 'problem solving',
        'technical troubleshooting'
    ],
    'call center': [
        'communication skills', 'phone etiquette', 'multitasking',
        'product knowledge', 'crm systems', 'patience'
    ],
    'sales': [
        'communication skills', 'persuasion', 'negotiation',
        'customer relations', 'product knowledge', 'closing techniques'
    ],
    'sales representative': [
        'communication skills', 'persuasion', 'negotiation',
        'customer relations', 'product knowledge', 'closing techniques'
    ],
    'bank': [
        'numerical accuracy', 'attention to detail', 'financial knowledge',
        'regulatory compliance', 'risk assessment', 'analytical thinking'
    ],
    'banking': [
        'numerical accuracy', 'attention to detail', 'financial knowledge',
        'regulatory compliance', 'risk assessment', 'analytical thinking'
    ],
    'financial': [
        'numerical accuracy', 'attention to detail', 'financial knowledge',
        'analytical thinking', 'data analysis', 'excel proficiency'
    ],
    'accountant': [
        'numerical accuracy', 'attention to detail', 'financial knowledge',
        'accounting principles', 'excel proficiency', 'analytical thinking'
    ],
    'data entry': [
        'typing speed', 'data entry accuracy', 'attention to detail',
        'office software proficiency', 'organizational skills'
    ],
    'entry level': [
        'basic computer literacy', 'communication skills', 'willingness to learn',
        'attention to detail', 'time management'
    ],
    'graduate': [
        'basic computer literacy', 'communication skills', 'willingness to learn',
        'analytical thinking', 'problem solving'
    ],
}


def expand_query_for_role(query, entities, expansion_weight=0.6):
    query_lower = query.lower()
    expanded_terms = []
    
    if entities.get('role'):
        role_lower = entities['role'].lower()
        for role_pattern, skills in ROLE_EXPANSION_MAP.items():
            if role_pattern in role_lower:
                expanded_terms.extend(skills)
                break
    
    if not expanded_terms:
        for role_pattern, skills in ROLE_EXPANSION_MAP.items():
            if role_pattern in query_lower:
                expanded_terms.extend(skills)
                break
    
    if not expanded_terms and entities.get('domain'):
        domain_lower = ' '.join(entities['domain']).lower()
        if 'admin' in domain_lower or 'administrative' in domain_lower:
            expanded_terms.extend(ROLE_EXPANSION_MAP.get('admin', []))
    
    if expanded_terms:
        expanded_terms = expanded_terms[:7]
        expansion_text = ' '.join(expanded_terms)
        return f"{query} {expansion_text}"
    
    return query


def get_expansion_terms_for_role(role):
    role_lower = role.lower()
    for role_pattern, skills in ROLE_EXPANSION_MAP.items():
        if role_pattern in role_lower:
            return skills
    return []
