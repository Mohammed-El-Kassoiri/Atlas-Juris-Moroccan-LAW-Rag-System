# FastAPI backend for RAG System
# This wraps the logic from stream.py into a REST API

import os, json, time, numpy as np, faiss, pickle, traceback, re, textwrap
from pathlib import Path
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Gemini SDK
try:
    from google import genai
except Exception:
    genai = None

# ===== CONFIG =====
# Get the project root directory (parent of backend/)
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
FLATTENED_JSONL = PROJECT_ROOT / "data" / "processed" / "flattened_docs.jsonl"
INDEX_FILE = PROJECT_ROOT / "law_index.faiss"
META_FILE = PROJECT_ROOT / "law_meta.pkl"
EMB_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
MAX_NEW_TOKENS = 512
DEVICE = "cpu"
# ==================

app = FastAPI(title="Legal RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- helpers for normalization -----------------
ARABIC_DIACRITICS_RE = re.compile(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]')
ARABIC_STOPWORDS = {"من","في","على","إلى","عن","ما","هو","هي","لم","لن","إن","أن","كل","قد","أو","و","التي","الذي","الذين","هذا","هذه","ذلك","تلك","مع","أنّ","إلا","كان","كانت","هناك","أي","سواء","بعد","قبل","حتى"}

def strip_diacritics_arabic(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text) if text else ""

def normalize_text_for_match(s: str) -> str:
    if not s:
        return ""
    s = strip_diacritics_arabic(s)
    s = s.lower()
    s = re.sub(r"[^\w\u0600-\u06FF]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_prefixes(token: str) -> str:
    if not token:
        return token
    t = token
    while t.startswith("و") and len(t) > 1:
        t = t[1:]
    if t.startswith("ال") and len(t) > 2:
        t = t[2:]
    if t.startswith("و") and len(t) > 1:
        t = t[1:]
    return t

def tokenize_and_clean(s: str):
    if not s:
        return []
    s_norm = normalize_text_for_match(s)
    tokens = [t for t in s_norm.split() if t]
    processed = []
    for t in tokens:
        t2 = strip_prefixes(t).strip()
        if not t2:
            continue
        if t2 in ARABIC_STOPWORDS:
            continue
        processed.append(t2)
    return processed

# ---------- Language detection ----------
ARABIC_CHAR_RE = re.compile(r'[\u0600-\u06FF]')
LATIN_CHAR_RE = re.compile(r'[A-Za-zÀ-ÖØ-öø-ÿ]')

def detect_language(s: str) -> str:
    if not s or not isinstance(s, str):
        return "other"
    ar_count = len(ARABIC_CHAR_RE.findall(s))
    lat_count = len(LATIN_CHAR_RE.findall(s))
    if ar_count > 0 and ar_count >= lat_count:
        return "ar"
    if lat_count > 0 and lat_count > ar_count:
        return "fr"
    return "other"

# --- Global state ---
_index = None
_texts = None
_metas = None
_embed_model = None

def prepare_index_and_meta():
    flat = FLATTENED_JSONL
    if not flat.exists():
        raise FileNotFoundError(f"{FLATTENED_JSONL} not found.")

    texts = []
    metas = []
    with flat.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text","")
            mada = obj.get("mada") or obj.get("id", f"m{i+1}")
            bab  = obj.get("bab") or obj.get("fasl") or ""
            source = obj.get("source") or bab or ""
            _id = obj.get("id") or f"{i+1:05d}"
            texts.append(text)
            lang = detect_language(text or mada or bab or source)
            metas.append({"id": _id, "mada": mada, "bab": bab, "source": source, "lang": lang})

    if INDEX_FILE.exists() and META_FILE.exists():
        try:
            index = faiss.read_index(str(INDEX_FILE))
            with open(META_FILE, "rb") as f:
                meta_pkl = pickle.load(f)
            if isinstance(meta_pkl, list) and len(meta_pkl) == len(texts):
                for i, m in enumerate(meta_pkl):
                    if "lang" not in m or not m.get("lang"):
                        meta_pkl[i]["lang"] = detect_language(texts[i] or m.get("mada") or m.get("bab") or m.get("source") or "")
                metas = meta_pkl
        except Exception:
            pass
    else:
        embedder = SentenceTransformer(EMB_MODEL_NAME)
        embs = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32)
        faiss.normalize_L2(embs)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        faiss.write_index(index, str(INDEX_FILE))
        with open(META_FILE, "wb") as f:
            pickle.dump(metas, f)
    return texts, metas, index

def load_index_and_embedder():
    global _index, _texts, _metas, _embed_model
    if _index is None or _texts is None:
        _texts, _metas, _index = prepare_index_and_meta()
        _embed_model = SentenceTransformer(EMB_MODEL_NAME)
    return _index, _texts, _metas, _embed_model

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    try:
        load_index_and_embedder()
        print("✅ Index and embedder loaded successfully")
    except Exception as e:
        print(f"❌ Error loading index: {e}")

# --- Retrieval helpers ---
def embed_query(text):
    _, _, _, embed_model = load_index_and_embedder()
    v = embed_model.encode([text], convert_to_numpy=True)
    if v.dtype != np.float32:
        v = v.astype(np.float32)
    faiss.normalize_L2(v)
    return v

def retrieve(query, top_k=TOP_K, prefer_same_language=True, strict_same_language=False):
    index, texts, metas, _ = load_index_and_embedder()
    q_lang = detect_language(query)
    qv = embed_query(query)
    D, I = index.search(qv, max(top_k * 6, top_k))
    candidates = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(texts):
            continue
        candidates.append({"score": float(score), "idx": int(idx), "text": texts[idx], "meta": metas[idx]})

    if strict_same_language and q_lang in ("ar", "fr"):
        same_lang = [c for c in candidates if c.get("meta", {}).get("lang") == q_lang]
        return sorted(same_lang, key=lambda x: x["score"], reverse=True)[:top_k]

    if prefer_same_language and q_lang in ("ar", "fr"):
        same_lang = [c for c in candidates if c.get("meta", {}).get("lang") == q_lang]
        if len(same_lang) >= top_k:
            return sorted(same_lang, key=lambda x: x["score"], reverse=True)[:top_k]
        selected = sorted(same_lang, key=lambda x: x["score"], reverse=True)
        others = [c for c in candidates if c.get("meta", {}).get("lang") != q_lang]
        others_sorted = sorted(others, key=lambda x: x["score"], reverse=True)
        selected.extend(others_sorted[: max(0, top_k - len(selected))])
        return selected

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

# ---------- Gemini helpers ----------
def build_instructional_prompt_from_retrieved(query, retrieved):
    context_parts = []
    for i, r in enumerate(retrieved, start=1):
        m = r.get("meta", {})
        mada = m.get("mada", "")
        bab = m.get("bab", "")
        src = m.get("source", "") or bab or ""
        text = r.get("text", "")
        # Show full text without truncation
        context_parts.append(f"المصدر {i}: ({mada} : {bab} : {src})\n{text}")

    context = "\n\n".join(context_parts)
    q_lang = detect_language(query)
    
    if q_lang == "fr":
        context = context.replace("المصدر ", "Source ")
        system_line = "SYSTEM: Vous êtes un avocat virtuel spécialisé en droit marocain."
        instr = textwrap.dedent(f"""\ 
        Reformulez les extraits suivants et produisez une réponse juridique unique et structurée — en français — comprenant, dans l'ordre :
        1) Un résumé bref (2-3 phrases).
        2) Une analyse juridique détaillée en s'appuyant exclusivement sur les extraits, en citant après chaque point (Article : Chapitre : Source).
        3) Une conclusion / conseil pratique court.
        4) Liste des références utilisées.

        Ne rajoutez pas d'informations extérieures aux extraits. Si les extraits sont insuffisants, indiquez-le clairement.

        Extraits:
        --------------------
        {context}
        --------------------

        Exigence : réponse organisée avec sous-titres (Résumé, Analyse juridique, Conclusion/Conseil, Références).
        """)
    else:
        system_line = "SYSTEM: أنت محامٍ افتراضي متخصص في القانون المغربي."
        instr = textwrap.dedent(f"""\ 
        أعد صياغة المقتطفات التالية وأنتج إجابة قانونية واحدة ومتكاملة — باللغة العربية الفصحى — وتتضمن بالترتيب:
        1) خلاصة موجزة (2-3 جمل).
        2) تحليل قانوني مفصّل يستند حصريًا إلى المقتطفات مع الإشارة بعد كل نقطة بالشكل (المادة : الباب : المصدر).
        3) استنتاج / نصيحة عملية قصيرة.
        4) قائمة المراجع المستخدمة.

        التزم بالمقتطفات ولا تضف معلومات خارجها. إن كانت المقتطفات غير كافية فاذكر ذلك صراحة.

        المقتطفات:
        --------------------
        {context}
        --------------------

        المطلوب: إجابة واحدة منظمة مع عناوين فرعية: (الخلاصة، التحليل القانوني، الاستنتاج/النصيحة العملية، المراجع).
        """)
    full_prompt = f"{system_line}\nQUESTION: {query}\n\n{instr}\nANSWER:\n"
    return full_prompt

def extract_text_from_gemini_response(resp) -> str:
    try:
        if hasattr(resp, "text"):
            txt = resp.text
            if callable(txt):
                txt = txt()
            if txt:
                return txt
    except Exception:
        pass

    try:
        candidates = getattr(resp, "candidates", None) or getattr(resp, "Candidates", None)
        if candidates:
            first = candidates[0]
            for attr in ("content", "Content"):
                cont = getattr(first, attr, None) or (first.get(attr) if isinstance(first, dict) else None)
                if cont:
                    parts = getattr(cont, "parts", None) or (cont.get("parts") if isinstance(cont, dict) else None)
                    if parts and len(parts) > 0:
                        p0 = parts[0]
                        if isinstance(p0, dict):
                            t = p0.get("text") or p0.get("Text")
                        else:
                            t = getattr(p0, "text", None) or getattr(p0, "Text", None)
                        if t:
                            return t
            if hasattr(first, "text"):
                t = first.text
                if callable(t):
                    t = t()
                if t:
                    return t
    except Exception:
        pass

    try:
        return str(resp)
    except Exception:
        return None

def call_gemini_generate(prompt, model_name="gemini-2.5-flash", max_output_tokens=None, temperature=0.0):
    api_key = "AIzaSyArOg9PSDtMQAOLAXERLdShaaSSxEnj_J8"
    if genai is None:
        raise RuntimeError("google-genai library not installed. pip install google-genai")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment. Set it before running.")

    client = genai.Client(api_key=api_key)
    kwargs = {"model": model_name, "contents": prompt}
    try:
        resp = client.models.generate_content(**kwargs)
    except Exception as e:
        raise

    text = extract_text_from_gemini_response(resp)
    return text

# --- Pydantic models ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    max_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.0
    model_name: str = "gemini-2.5-flash"
    prefer_same_language: bool = True
    strict_same_language: bool = False
    include_prompt: bool = False

class QueryResponse(BaseModel):
    answer: str
    retrieved: List[Dict[str, Any]]
    prompt: Optional[str] = None
    total_time: float
    retrieval_time: float
    generation_time: float
    query_lang: str

class StatsResponse(BaseModel):
    total_snippets: int

# --- API endpoints ---
@app.get("/")
async def root():
    return {"message": "Legal RAG API", "status": "running"}

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    _, texts, _, _ = load_index_and_embedder()
    return StatsResponse(total_snippets=len(texts))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        t0 = time.time()
        retrieved = retrieve(
            request.query,
            top_k=request.top_k,
            prefer_same_language=request.prefer_same_language,
            strict_same_language=request.strict_same_language
        )
        retrieval_time = time.time() - t0

        if not retrieved:
            raise HTTPException(status_code=404, detail="No relevant texts found.")

        prompt = build_instructional_prompt_from_retrieved(request.query, retrieved)
        
        gen_start = time.time()
        try:
            generated = call_gemini_generate(
                prompt,
                model_name=request.model_name,
                temperature=float(request.temperature)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        
        generation_time = time.time() - gen_start
        total_time = time.time() - t0
        q_lang = detect_language(request.query)

        return QueryResponse(
            answer=generated or "No response generated.",
            retrieved=retrieved,
            prompt=prompt if request.include_prompt else None,
            total_time=total_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            query_lang=q_lang
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

