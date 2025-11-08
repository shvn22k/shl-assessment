from urllib.parse import urlparse
import re

def canonical_path(url):
    if not url:
        return ""
    p = urlparse(url)
    path = (p.path or "").strip().lower()
    path = re.sub(r"/+", "/", path).strip("/")
    path = re.sub(r"^solutions/products/", "products/", path)
    path = re.sub(r"^solutions/products/product-catalog/", "products/product-catalog/", path)
    path = re.sub(r"^solutions/", "", path)
    path = re.sub(r"-\d+(-\d+)*", "", path)
    path = re.sub(r"-new\b", "", path)
    path = re.sub(r"\.html?$", "", path)
    return path
