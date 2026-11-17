"""Wikipedia API yardımcıları (MediaWiki Action API)."""
import re
import requests

HEADERS = {"User-Agent": "WikiArastirmaBot/1.0 (egitim amacli proje)"}
TIMEOUT = 15


def _api(lang: str) -> str:
    lang = lang if re.fullmatch(r"[a-z\-]{2,12}", lang or "") else "tr"
    return f"https://{lang}.wikipedia.org/w/api.php"


def _clean(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").replace("&quot;", '"').replace("&amp;", "&")


def search(query: str, lang: str = "tr", limit: int = 6) -> list[dict]:
    """Wikipedia'da arama yapar; başlık, kısa özet ve kelime sayısı döner."""
    r = requests.get(_api(lang), headers=HEADERS, timeout=TIMEOUT, params={
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": max(1, min(limit, 10)), "format": "json", "formatversion": 2,
    })
    r.raise_for_status()
    hits = r.json().get("query", {}).get("search", [])
    return [{"title": h["title"], "snippet": _clean(h.get("snippet", "")),
             "words": h.get("wordcount", 0)} for h in hits]


def get_article(title: str, lang: str = "tr", full: bool = False, max_chars: int = 12000) -> dict:
    """Bir makalenin metnini (giriş veya tamamı), URL'sini ve küçük resmini döner."""
    params = {
        "action": "query", "titles": title, "redirects": 1, "format": "json", "formatversion": 2,
        "prop": "extracts|info|pageimages|description", "explaintext": 1,
        "inprop": "url", "piprop": "thumbnail", "pithumbsize": 320,
    }
    if not full:
        params["exintro"] = 1
    r = requests.get(_api(lang), headers=HEADERS, timeout=TIMEOUT, params=params)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return {"found": False, "title": title}
    p = pages[0]
    text = p.get("extract", "") or ""
    truncated = len(text) > max_chars
    return {
        "found": True,
        "title": p.get("title", title),
        "description": p.get("description", ""),
        "url": p.get("fullurl", f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"),
        "thumbnail": (p.get("thumbnail") or {}).get("source"),
        "text": text[:max_chars] + ("\n...[metin kısaltıldı]" if truncated else ""),
    }


def random_titles(lang: str = "tr", n: int = 4) -> list[str]:
    r = requests.get(_api(lang), headers=HEADERS, timeout=TIMEOUT, params={
        "action": "query", "list": "random", "rnnamespace": 0, "rnlimit": n,
        "format": "json", "formatversion": 2,
    })
    r.raise_for_status()
    return [x["title"] for x in r.json().get("query", {}).get("random", [])]
