from urllib.parse import urlparse
import re

def canonical_path(url):
    if not url:
        return ""
    
    p = urlparse(url)
    path = (p.path or "").strip().lower()
    
    # normalize slashes
    path = re.sub(r"/+", "/", path).strip("/")
    
    # fix common prefixes
    path = re.sub(r"^solutions/products/", "products/", path)
    path = re.sub(r"^solutions/products/product-catalog/", "products/product-catalog/", path)
    path = re.sub(r"^solutions/", "", path)
    
    # remove version numbers and suffixes
    path = re.sub(r"-\d+(-\d+)*", "", path)
    path = re.sub(r"-new\b", "", path)
    path = re.sub(r"\.html?$", "", path)
    
    return path
