from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def build_tfidf_index(docs):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, stop_words='english')
    X = vec.fit_transform(docs)
    return vec, X

def tfidf_scores(vec, X, query_text):
    qv = vec.transform([query_text])
    sims = (qv @ X.T).toarray()[0]
    # normalize
    if sims.max() > 0:
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-12)
    return sims

def fuse_scores(embed_scores, tfidf_scores, w_embed=0.65, w_tfidf=0.35):
    e = np.array(embed_scores)
    t = np.array(tfidf_scores)
    
    # normalize both
    if e.max() > 0:
        e = (e - e.min()) / (e.max() - e.min() + 1e-12)
    if t.max() > 0:
        t = (t - t.min()) / (t.max() - t.min() + 1e-12)
    
    # combine
    fused = w_embed * e + w_tfidf * t
    return fused.tolist()
