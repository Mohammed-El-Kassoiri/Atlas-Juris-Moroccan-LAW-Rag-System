# streamlit_app.py
# Streamlit UI for Local RAG System using local embeddings + FAISS and Gemini (Google) generation
# Requirements:
#   pip install streamlit sentence-transformers faiss-cpu google-genai
# Run:
#   streamlit run streamlit_app.py

import streamlit as st
import os, json, time, numpy as np, faiss, pickle, traceback, re, textwrap
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Gemini SDK
try:
    from google import genai
except Exception:
    genai = None

# ===== CONFIG (edit these paths if needed) =====
FLATTENED_JSONL = r"data/processed/flattened_docs.jsonl"
INDEX_FILE = r"law_index.faiss"
META_FILE = r"law_meta.pkl"
EMB_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # embedding model

TOP_K = 3
MAX_NEW_TOKENS = 512
DEVICE = "cpu"
# ===============================================

st.set_page_config(page_title="محامي افتراضي - قوانين المغرب", page_icon="⚖️", layout="wide")
st.title("⚖️ محامي افتراضي محلي — قوانين المغرب")
st.caption("🔧 يعمل محليًا للاسترجاع، ويستخدم Gemini للتوليد (ضع GEMINI_API_KEY كمتغير بيئي)")

# ----------------- helpers for normalization (for metadata matching) -----------------
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

def token_overlap_score(query_norm: str, meta_norm: str) -> float:
    if not query_norm or not meta_norm:
        return 0.0
    q_toks = set(tokenize_and_clean(query_norm))
    m_toks = set(tokenize_and_clean(meta_norm))
    if not q_toks or not m_toks:
        return 0.0
    shared = q_toks.intersection(m_toks)
    return len(shared) / max(1, len(q_toks))

# ---------- Simple language detection (heuristic) ----------
ARABIC_CHAR_RE = re.compile(r'[\u0600-\u06FF]')
LATIN_CHAR_RE = re.compile(r'[A-Za-zÀ-ÖØ-öø-ÿ]')

def detect_language(s: str) -> str:
    """Very simple heuristic: returns 'ar' for Arabic-heavy text, 'fr' for Latin-heavy,
    otherwise 'other'."""
    if not s or not isinstance(s, str):
        return "other"
    ar_count = len(ARABIC_CHAR_RE.findall(s))
    lat_count = len(LATIN_CHAR_RE.findall(s))
    # bias threshold: prefer Arabic if Arabic chars >= Latin chars
    if ar_count > 0 and ar_count >= lat_count:
        return "ar"
    if lat_count > 0 and lat_count > ar_count:
        return "fr"
    return "other"

# --- prepare index & metas (build if missing) ---
def prepare_index_and_meta():
    flat = Path(FLATTENED_JSONL)
    if not flat.exists():
        raise FileNotFoundError(f"{FLATTENED_JSONL} not found. Put your flattened_docs.jsonl at that path.")

    texts = []
    metas = []
    with flat.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text","")
            # ensure metadata fields exist
            mada = obj.get("mada") or obj.get("id", f"m{i+1}")
            bab  = obj.get("bab") or obj.get("fasl") or ""
            source = obj.get("source") or bab or ""
            _id = obj.get("id") or f"{i+1:05d}"
            texts.append(text)
            # detect language of this snippet (fallback to source/title too)
            lang = detect_language(text or mada or bab or source)
            metas.append({"id": _id, "mada": mada, "bab": bab, "source": source, "lang": lang})

    # if index and meta pickle exist, load them to speed up
    if Path(INDEX_FILE).exists() and Path(META_FILE).exists():
        try:
            # validate meta file; ensure 'lang' exists for each meta (backwards compatibility)
            index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as f:
                meta_pkl = pickle.load(f)
            if isinstance(meta_pkl, list) and len(meta_pkl) == len(texts):
                # ensure each meta has 'lang'
                for i, m in enumerate(meta_pkl):
                    if "lang" not in m or not m.get("lang"):
                        meta_pkl[i]["lang"] = detect_language(texts[i] or m.get("mada") or m.get("bab") or m.get("source") or "")
                metas = meta_pkl
        except Exception:
            # fall through to rebuild embeddings if anything goes wrong
            pass
    else:
        st.info("بناء الفهرس (مرة واحدة) — قد يستغرق بعض الوقت...")
        embedder = SentenceTransformer(EMB_MODEL_NAME)
        embs = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32)
        faiss.normalize_L2(embs)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(metas, f)
        st.success("تم بناء الفهرس وحفظه.")
    return texts, metas

# --- load index & embed model (cached) ---
@st.cache_resource
def load_index_and_embedder():
    texts, metas = prepare_index_and_meta()
    index = faiss.read_index(INDEX_FILE)
    embed_model = SentenceTransformer(EMB_MODEL_NAME)
    return index, texts, metas, embed_model

with st.spinner("⏳ جاري تحميل الفهرس ونموذج التضمين..."):
    index, texts, metas, embed_model = load_index_and_embedder()
st.success(f"✅ الفهرس جاهز — مقتطفات: {len(texts)}")

# --- retrieval helpers ---
def embed_query(text):
    v = embed_model.encode([text], convert_to_numpy=True)
    if v.dtype != np.float32:
        v = v.astype(np.float32)
    faiss.normalize_L2(v)
    return v

def retrieve(query, top_k=TOP_K, prefer_same_language=True, strict_same_language=False):
    """
    Retrieve top_k candidates, optionally preferring or strictly filtering to same-language snippets.
    If strict_same_language is True, only returns docs where meta.lang == detected query language
    (may return fewer than top_k if not enough matches).
    """
    q_lang = detect_language(query)
    qv = embed_query(query)
    # retrieve a larger candidate pool to allow filtering/re-ranking
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
            selected = sorted(same_lang, key=lambda x: x["score"], reverse=True)[:top_k]
            return selected
        # otherwise take as many same-lang as possible then fill with best others
        selected = sorted(same_lang, key=lambda x: x["score"], reverse=True)
        others = [c for c in candidates if c.get("meta", {}).get("lang") != q_lang]
        others_sorted = sorted(others, key=lambda x: x["score"], reverse=True)
        selected.extend(others_sorted[: max(0, top_k - len(selected))])
        return selected

    # default: just top_k by score
    out = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
    return out

# ---------- Gemini helpers ----------
def build_instructional_prompt_from_retrieved(query, retrieved):
    """Create the single combined prompt instructing Gemini to produce the structured legal answer.
       The prompt language matches the detected query language (ar/fr)."""
    # Build a context with numbered sources
    context_parts = []
    for i, r in enumerate(retrieved, start=1):
        m = r.get("meta", {})
        mada = m.get("mada", "")
        bab = m.get("bab", "")
        src = m.get("source", "") or bab or ""
        text = r.get("text", "")
        snippet = text if len(text) <= 4000 else text[:4000] + " ...[truncated]"
        # localize source label depending on expected language -- default Arabic label used below, replaced for French if needed
        context_parts.append(f"المصدر {i}: ({mada} : {bab} : {src})\n{snippet}")

    context = "\n\n".join(context_parts)

    q_lang = detect_language(query)
    if q_lang == "fr":
        # replace Arabic "المصدر" labels with French ones in the context
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
        # default to Arabic
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
    """Robust extractor for various SDK response shapes."""
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
            # many shapes: try common traversals
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

    # fallback: return stringified object for debugging
    try:
        return str(resp)
    except Exception:
        return None

def call_gemini_generate(prompt, model_name="gemini-2.5-flash", max_output_tokens=None, temperature=0.0):
    """Call Gemini (google-genai). Returns generated text or raises error."""
    api_key = "AIzaSyArOg9PSDtMQAOLAXERLdShaaSSxEnj_J8"  # expect user to set this in environment
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

# --- Streamlit UI state & controls ---
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ الإعدادات")
    top_k = st.slider("عدد المقتطفات المسترجعة", 1, 8, TOP_K)
    max_tokens = st.slider("طول الإجابة (حد أقصى) تقريبًا", 128, 2048, MAX_NEW_TOKENS, step=64)
    temp = st.slider("درجة العشوائية (temperature)", 0.0, 1.0, 0.0, step=0.05)
    model_choice = st.selectbox("نموذج التوليد (Gemini)", ["gemini-2.5-flash"], index=0)
    show_prompt = st.checkbox("إظهار الطلب (prompt) المرسل للنموذج (Debug)", value=False)
    st.divider()
    st.subheader("خيارات التحكم في اللغة")
    prefer_same_lang = st.checkbox("فضّل مقتطفات بنفس لغة السؤال (افتراضي)", value=True)
    strict_same_lang = st.checkbox("التصفية الصارمة: إرجاع مقتطفات من نفس اللغة فقط (قد يعيد أقل من المطلوب)", value=False)
    st.divider()
    st.subheader("📊 الإحصائيات")
    st.metric("المقتطفات المتاحة", f"{len(texts):,}")
    if st.button("🗑️ مسح السجل"):
        st.session_state.history = []
        st.experimental_rerun()

# Main input
col1, col2 = st.columns([4,1])
with col1:
    query = st.text_input("اكتب سؤالك القانوني بالعربية أو بالفرنسية:", placeholder="مثال (عربية): ما هي مهام مؤسسة ...؟ — مثال (français): Quelles sont les règles du cautionnement de comparution ?")
with col2:
    ask_button = st.button("🔍 اسأل", use_container_width=True)

if ask_button and query:
    with st.spinner("⏳ جاري الاسترجاع والتوليد..."):
        t0 = time.time()
        retrieved = retrieve(query, top_k=top_k, prefer_same_language=prefer_same_lang, strict_same_language=strict_same_lang)
        retrieval_time = time.time() - t0

        if not retrieved:
            st.error("❌ لم يتم العثور على نصوص ذات صلة.")
        else:
            prompt = build_instructional_prompt_from_retrieved(query, retrieved)
            if show_prompt:
                st.subheader("🔎 Prompt (sent to Gemini)")
                st.code(prompt[:4000] + ("\n\n...[truncated]" if len(prompt) > 4000 else ""), language="text")

            gen_start = time.time()
            try:
                generated = call_gemini_generate(prompt, model_name=model_choice, temperature=float(temp))
            except Exception as e:
                st.error(f"فشل التوليد عبر Gemini: {e}")
                st.write(traceback.format_exc())
                generated = None
            generation_time = time.time() - gen_start
            total_time = time.time() - t0

            if generated:
                st.session_state.history.append({
                    "query": query,
                    "answer": generated,
                    "retrieved": retrieved,
                    "time": total_time,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time
                })

# Display history
if st.session_state.history:
    st.divider()
    st.subheader("📝 سجل الأسئلة والأجوبة")
    for i, item in enumerate(reversed(st.session_state.history), start=1):
        with st.container():
            st.markdown(f"### ❓ {item['query']}")
            st.success(item["answer"])
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("⏱️ الوقت الإجمالي", f"{item['time']:.2f}s")
            with col_b:
                st.metric("🔍 استرجاع", f"{item['retrieval_time']:.2f}s")
            with col_c:
                st.metric("🤖 توليد", f"{item['generation_time']:.2f}s")

            with st.expander("📚 المصادر المستخدمة"):
                for j, r in enumerate(item["retrieved"], start=1):
                    m = r.get("meta", {})
                    mada = m.get("mada", "")
                    bab  = m.get("bab", "")
                    src  = m.get("source", "") or bab
                    lang = m.get("lang", "other")
                    st.markdown(f"- ({mada} : {bab} : {src}) — لغة: {lang} — تشابه: {r['score']:.3f}")
                    st.markdown(f"> {r['text'][:600]}...")

            st.divider()
else:
    st.info("👆 اكتب سؤالك أعلاه للبدء")

st.divider()
st.caption("🤖 RAG محلي + Gemini للتوليد | اضبط GEMINI_API_KEY في بيئتك قبل التشغيل")