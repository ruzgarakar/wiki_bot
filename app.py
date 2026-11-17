"""Wiki Araştırma Asistanı — Flask sunucusu.

Kullanıcı doğal dilde soru sorar; dil modeli soruyu anlar, Wikipedia API'yi araç olarak
kullanarak ilgili makaleleri arar/okur ve açıklayıcı bir özet yazar.
İlerleme (aranan terimler, okunan makaleler) tarayıcıya NDJSON akışı olarak gönderilir.

Ücretsiz kullanılabilen, OpenAI uyumlu herhangi bir servisle çalışır:
Google Gemini, Groq, OpenRouter veya bilgisayarınızda çalışan Ollama.
"""
import json
import os
import re
import time
from pathlib import Path

import openai
from flask import Flask, Response, jsonify, request, send_from_directory

import wikipedia as wiki

# --- .env dosyasını yükle (varsa) -------------------------------------------
_env = Path(__file__).with_name(".env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# --- Ücretsiz sağlayıcı ön ayarları ------------------------------------------
PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-2.5-flash",
               "key_url": "https://aistudio.google.com/apikey"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b",
             "key_url": "https://console.groq.com/keys"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.3-70b-instruct:free",
                   "key_url": "https://openrouter.ai/keys"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b", "key_url": None},
}

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
_preset = PROVIDERS.get(PROVIDER, {})
BASE_URL = os.environ.get("LLM_BASE_URL") or _preset.get("base_url")
MODEL = os.environ.get("LLM_MODEL") or _preset.get("model", "")
API_KEY = os.environ.get("LLM_API_KEY", "") or ("ollama" if PROVIDER == "ollama" else "")
MAX_STEPS = 6

# Groq'un ücretsiz planında dakikada ~8000 token sınırı var; makaleleri daha kısa okuyup
# cevap uzunluğunu sınırlıyoruz. Diğer sağlayıcılar daha geniş bütçe kullanır.
BUDGET = ({"intro": 2500, "full": 5000, "max_tokens": 2048} if PROVIDER == "groq"
          else {"intro": 6000, "full": 12000, "max_tokens": 4096})

app = Flask(__name__, static_folder="static")
# Hız sınırı (429) beklemelerini kendimiz yönetiyoruz ki kullanıcıya gösterebilelim.
client = openai.OpenAI(api_key=API_KEY or "eksik", base_url=BASE_URL, timeout=120, max_retries=0)

LANG_NAMES = {"tr": "Türkçe", "en": "English", "de": "Deutsch", "fr": "Français",
              "es": "Español", "ar": "العربية", "ru": "Русский"}

DETAIL = {
    "kisa": "Kısa ve öz yaz: 1-2 paragraf, en fazla ~150 kelime.",
    "orta": "Dengeli bir anlatım yaz: birkaç paragraf ve gerekirse madde işaretleri, ~300-450 kelime.",
    "detayli": "Kapsamlı yaz: başlıklar (##), alt bölümler, önemli tarih/kavramlar ve örneklerle, ~700-1000 kelime.",
}

SYSTEM = """Sen bir Wikipedia araştırma asistanısın. Kullanıcı sana doğal bir dille, bazen belirsiz ya da günlük konuşma diliyle soru sorar.

Görevin:
1. Kullanıcının asıl ne öğrenmek istediğini anla (yazım hatalarını, belirsiz ifadeleri, dolaylı soruları yorumla).
2. `wiki_search` aracıyla ilgili makaleleri bul; gerekirse farklı anahtar kelimelerle birden fazla arama yap. Soru birden çok konuya dokunuyorsa her biri için ayrı ara.
3. `wiki_read` aracıyla en alakalı makaleleri oku (genelde 2-4 makale). Giriş kısmı yetmezse full=true ile tamamını oku.
4. Okuduklarını sentezleyerek kullanıcıya açıklayıcı, anlaşılır, öğretici bir anlatımla cevap ver.

Kurallar:
- Cevap yazmadan önce MUTLAKA en az bir kez arama yapıp en az bir makale oku.
- Bilgileri Wikipedia'dan okuduklarına dayandır; makalelerde olmayan bir şeyi kesinmiş gibi sunma. Wikipedia'da bulunamayan kısımları açıkça belirt.
- Cevabı Markdown ile biçimlendir (kalın, listeler, gerekirse ## başlıklar). Bilgi aldığın makaleye metin içinde [Makale Adı] şeklinde atıf yapabilirsin.
- Teknik terimleri sade bir dille açıkla; gerekirse benzetme ve örnek kullan.
- Cevabın EN SONUNA, kullanıcının merak edebileceği 3 ilgili konuyu tam olarak şu biçimde ekle:
<ilgili>Konu 1 | Konu 2 | Konu 3</ilgili>"""

TOOLS = [
    {"type": "function", "function": {
        "name": "wiki_search",
        "description": "Wikipedia'da anahtar kelimelerle arama yapar. Eşleşen makale başlıklarını ve kısa özetlerini döner.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Kısa, isabetli arama terimi, örn. 'Osmanlı İmparatorluğu kuruluşu'"},
                "lang": {"type": "string", "description": "Wikipedia dil kodu (tr, en, de...). Kullanıcının seçtiği dili kullan; kaynak yetersizse 'en' dene."},
            },
            "required": ["query", "lang"],
        },
    }},
    {"type": "function", "function": {
        "name": "wiki_read",
        "description": "Bir Wikipedia makalesinin metnini okur. full=false sadece giriş bölümünü, full=true makalenin tamamını getirir.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Makalenin tam başlığı (wiki_search sonucundan)"},
                "lang": {"type": "string", "description": "Wikipedia dil kodu"},
                "full": {"type": "boolean", "description": "Makalenin tamamı mı okunsun?"},
            },
            "required": ["title", "lang"],
        },
    }},
]


def run_tool(name: str, args: dict, sources: dict, default_lang: str) -> tuple[str, dict]:
    """Aracı çalıştırır; (modelin göreceği sonuç metni, arayüze gidecek olay) döner."""
    lang = args.get("lang") or default_lang
    if name == "wiki_search":
        hits = wiki.search(str(args.get("query", "")), lang)
        result = json.dumps(hits, ensure_ascii=False) if hits else "Sonuç bulunamadı. Farklı kelimelerle veya başka dilde dene."
        return result, {"type": "search", "query": args.get("query"), "lang": lang,
                        "results": [h["title"] for h in hits]}
    if name == "wiki_read":
        full = str(args.get("full")).lower() == "true"
        art = wiki.get_article(str(args.get("title", "")), lang, full,
                               max_chars=BUDGET["full"] if full else BUDGET["intro"])
        if not art["found"]:
            return f"'{args.get('title')}' başlıklı makale bulunamadı.", {"type": "read", "title": args.get("title"), "found": False}
        key = art["url"]
        if key not in sources:
            sources[key] = {k: art[k] for k in ("title", "description", "url", "thumbnail")}
        result = f"# {art['title']}\n{art['description']}\n\n{art['text']}"
        return result, {"type": "read", "found": True, "full": full, **sources[key]}
    return f"Bilinmeyen araç: {name}", {"type": "status", "message": f"Bilinmeyen araç: {name}"}


def split_related(text: str) -> tuple[str, list[str]]:
    m = re.search(r"<ilgili>(.*?)</ilgili>", text, re.S)
    if not m:
        return text.strip(), []
    topics = [t.strip() for t in m.group(1).split("|") if t.strip()]
    return (text[:m.start()] + text[m.end():]).strip(), topics[:5]


def _retry_after(err: Exception, default: int) -> int:
    """Hata yanıtındaki retry-after başlığından bekleme süresini (saniye) çıkarır."""
    try:
        val = float(err.response.headers.get("retry-after"))
        return max(1, min(60, int(val + 0.999)))
    except Exception:
        return default


def call_model(messages: list, allow_tools: bool):
    kwargs = dict(model=MODEL, messages=messages, max_tokens=BUDGET["max_tokens"], temperature=0.3)
    if allow_tools:
        kwargs.update(tools=TOOLS, tool_choice="auto")
    try:
        return client.chat.completions.create(**kwargs)
    except openai.BadRequestError as e:
        # Bazı ücretsiz modeller bazen bozuk araç çağrısı üretir (ör. Groq "tool_use_failed").
        # Bir kez daha dene; yine olmazsa araçsız devam et.
        if not allow_tools:
            raise
        try:
            return client.chat.completions.create(**kwargs)
        except openai.BadRequestError:
            kwargs.pop("tools"); kwargs.pop("tool_choice")
            messages.append({"role": "user", "content": "(Araç çağrısı başarısız oldu. Şimdiye kadar elde ettiğin bilgilerle cevabı yaz.)"})
            return client.chat.completions.create(**kwargs)


def research(question: str, lang: str, detail: str, history: list[dict]):
    """Araştırma döngüsü — her adımda bir olay (dict) üretir."""
    sources: dict = {}
    messages: list = [{"role": "system", "content": SYSTEM}]
    for h in history[-6:]:  # önceki soru/cevaplar (takip soruları için bağlam)
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:6000]})
    messages.append({"role": "user", "content": (
        f"{question}\n\n---\n(Tercih edilen Wikipedia dili: {lang} / {LANG_NAMES.get(lang, lang)}. "
        f"Cevabı kullanıcının soruyu sorduğu dilde yaz. Ayrıntı düzeyi: {DETAIL.get(detail, DETAIL['orta'])})"
    )})

    yield {"type": "status", "message": "Soru analiz ediliyor..."}

    for step in range(MAX_STEPS + 1):
        last = step == MAX_STEPS
        if last:
            messages.append({"role": "user", "content": "(Araştırma için ayrılan adım sayısı doldu. Topladığın bilgilerle cevabı şimdi yaz.)"})
        for attempt in range(4):
            try:
                response = call_model(messages, allow_tools=not last)
                break
            except (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError) as e:
                if attempt == 3:
                    raise
                wait = _retry_after(e, default=5 * (attempt + 1))
                if isinstance(e, openai.RateLimitError):
                    yield {"type": "status", "message": f"⏳ Dakikalık ücretsiz sınıra ulaşıldı, {wait} sn bekleyip devam edilecek (cevap gelecek)..."}
                else:
                    yield {"type": "status", "message": f"Servis yanıt vermedi, {wait} sn sonra tekrar denenecek..."}
                time.sleep(wait)
        choice = response.choices[0]
        msg = choice.message
        tool_calls = msg.tool_calls or []

        if tool_calls and not last:
            messages.append(msg.model_dump(exclude_none=True))
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}
                try:
                    out, event = run_tool(tc.function.name, args, sources, lang)
                    yield event
                except Exception as e:  # ağ hatası vb.
                    out = f"Hata: {e}"
                    yield {"type": "status", "message": f"Wikipedia isteği başarısız: {e}"}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            yield {"type": "status", "message": "Bilgiler değerlendiriliyor..."}
            continue

        text = msg.content or ""
        if not text.strip():
            yield {"type": "error", "message": "Model boş bir cevap döndürdü. Lütfen tekrar deneyin."}
            return
        if choice.finish_reason == "length":
            text += "\n\n*(Cevap uzunluk sınırına ulaştığı için kesildi.)*"
        answer, related = split_related(text)
        yield {"type": "answer", "text": answer, "related": related,
               "sources": list(sources.values()), "model": response.model or MODEL}
        return


def _line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


@app.post("/api/ask")
def ask():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Soru boş olamaz."}), 400
    lang = data.get("lang", "tr")
    detail = data.get("detail", "orta")
    history = data.get("history") or []

    def stream():
        if not API_KEY:
            yield _line({"type": "error", "message": "Dil modeli API anahtarı eksik. .env dosyasına LLM_API_KEY ekleyip sunucuyu yeniden başlatın."})
            return
        try:
            for event in research(question[:2000], lang, detail, history):
                yield _line(event)
        except openai.AuthenticationError:
            yield _line({"type": "error", "message": "API anahtarı geçersiz. .env dosyasındaki LLM_API_KEY değerini kontrol edin."})
        except openai.RateLimitError:
            yield _line({"type": "error", "message": "Ücretsiz kullanım sınırı aşıldı, bu soru cevaplanamadı. Birkaç dakika sonra tekrar deneyin; hâlâ olmuyorsa günlük sınır dolmuştur ve yarın yenilenir."})
        except openai.NotFoundError:
            yield _line({"type": "error", "message": f"'{MODEL}' modeli bulunamadı. .env dosyasında LLM_MODEL değerini değiştirin."})
        except openai.APIConnectionError:
            hint = " Ollama açık mı?" if PROVIDER == "ollama" else " İnternet bağlantınızı kontrol edin."
            yield _line({"type": "error", "message": "Dil modeli servisine bağlanılamadı." + hint})
        except openai.APIStatusError as e:
            yield _line({"type": "error", "message": f"API hatası ({e.status_code}): {e.message}"})
        except Exception as e:
            yield _line({"type": "error", "message": f"Beklenmeyen hata: {e}"})

    return Response(stream(), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/random")
def random_topics():
    try:
        return jsonify(wiki.random_titles(request.args.get("lang", "tr")))
    except Exception:
        return jsonify([])


@app.get("/api/status")
def status():
    return jsonify({"provider": PROVIDER, "model": MODEL, "api_key": bool(API_KEY),
                    "key_url": _preset.get("key_url")})


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Wiki Araştırma Asistanı: http://localhost:{port}  (sağlayıcı: {PROVIDER}, model: {MODEL})")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
