# app.py — Classroom Co-Pilot AI (Enhanced Edition)
# ─────────────────────────────────────────────────────────────────────────────
# Production-ready Streamlit application.
#
# Architecture (Multipage via st.navigation):
#   - Teacher Command Center (Page 1)
#   - Student Learning Board (Page 2)
#   - Session Log (Page 3)
# ─────────────────────────────────────────────────────────────────────────────

import os
import hashlib
import tempfile
import streamlit as st
import streamlit.components.v1 as components

import stt
import llm
import tts
import utils
import doc_processor
import retriever

# ─── ENV ──────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env")
    load_dotenv(_env if os.path.exists(_env) else None)
except ImportError:
    pass

def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets (Streamlit Cloud) first, then os.environ (local)."""
    try:
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MentorAI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
_css = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
if os.path.exists(_css):
    with open(_css, "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
_GEMINI_KEY     = _get_secret("GEMINI_API_KEY")
_OPENROUTER_KEY = _get_secret("OPENROUTER_API_KEY")

_DEFAULTS = {
    "history":          [],
    "transcript":       "",
    "content":          None,
    "last_audio_hash":  None,
    "revealed":         False,
    "speak_text":       None,
    "class_level":      "Class 6",
    "subject":          "",
    "chapter":          "",
    "mute_tts":         False,
    "ollama_model":     "llama3",
    "provider":         "Gemini" if _GEMINI_KEY else ("OpenRouter" if _OPENROUTER_KEY else "Ollama"),
    "gemini_api_key":   _GEMINI_KEY,
    "openrouter_api_key": _OPENROUTER_KEY,
    "openrouter_model": "openrouter/free",
    "doc_store":        doc_processor.DocumentStore(),
    "socratic_chat":    [],
    "last_socratic_audio_hash": None,
    "debate_chat":      [],
    "last_debate_audio_hash": None,
    "topic_input_value": "",
    "content_mode":     "NORMAL",  # NORMAL | SOCRATIC | DEBATE
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════════
# CORE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def process_command(cmd: str) -> bool:
    """Full pipeline: command → RAG → LLM (Gemini/OpenRouter/Ollama) → content → TTS queue."""
    if not cmd.strip():
        return False

    # Check configurations based on selected provider
    if st.session_state.provider == "Gemini":
        model_name = "gemini-2.5-flash"
        api_key = st.session_state.gemini_api_key
        if not api_key:
            st.error("⚠️ Gemini API Key nahi mila — sidebar mein **AI Configuration** mein enter karein.")
            return False
    elif st.session_state.provider == "OpenRouter":
        model_name = st.session_state.openrouter_model
        api_key = st.session_state.openrouter_api_key
        if not api_key:
            st.error("⚠️ OpenRouter API Key nahi mila — sidebar mein **AI Configuration** mein enter karein.")
            return False
    else:
        model_name = st.session_state.ollama_model
        api_key = ""
        if not model_name:
            st.error("⚠️ Ollama Model name nahi mila — sidebar mein **AI Configuration** mein enter karein.")
            return False

    topic = utils.extract_topic(cmd)
    if topic:
        st.session_state.topic_input_value = topic
    
    # RAG Retrieval
    context_str = ""
    chunks_found = 0
    if not st.session_state.doc_store.is_empty():
        with st.spinner("📚 Searching textbook..."):
            retrieved = retriever.search(
                query=cmd,
                doc_store=st.session_state.doc_store,
                topic=topic,
                chapter=st.session_state.chapter,
                subject=st.session_state.subject,
                top_k=4
            )
            chunks_found = len(retrieved)
            context_str = retriever.build_context_string(retrieved)

    with st.spinner("✨ MentorAI soch raha hai…"):
        try:
            content = llm.generate_classroom_content(
                provider=st.session_state.provider,
                model_name=model_name,
                api_key=api_key,
                command=cmd,
                class_level=st.session_state.class_level,
                context=context_str,
                subject=st.session_state.subject,
                chapter=st.session_state.chapter,
                topic=topic
            )
            content._rag_chunks = chunks_found  # store dynamically for display
        except Exception as exc:
            st.error(f"❌ {exc}")
            return False

    st.session_state.content  = content
    st.session_state.revealed = False
    st.session_state.history.append({"command": cmd, "content": content})

    # Determine mode STRICTLY from the command text — never from AI response content
    if "debate" in cmd.lower():
        st.session_state.content_mode = "DEBATE"
    elif "socratic" in cmd.lower() or "think" in cmd.lower():
        st.session_state.content_mode = "SOCRATIC"
    else:
        st.session_state.content_mode = "NORMAL"

    if st.session_state.content_mode == "DEBATE":
        st.session_state.debate_chat = [
            {"role": "assistant", "text": "Aap kis perspective ke paksh mein hain (Perspective A ya Perspective B)? Apna point of view (argument) yahan likhein ya bolein, aur chaliye charcha shuru karte hain!"}
        ]
        st.session_state.last_debate_audio_hash = None
        st.session_state.socratic_chat = []
    elif st.session_state.content_mode == "SOCRATIC":
        sq = getattr(content, "socratic_question", None) or "Aapko kya lagta hai is topic ke baare mein?"
        st.session_state.socratic_chat = [
            {"role": "assistant", "text": sq}
        ]
        st.session_state.last_socratic_audio_hash = None
        st.session_state.debate_chat = []
    else:
        st.session_state.socratic_chat = []
        st.session_state.debate_chat = []

    if not st.session_state.mute_tts:
        st.session_state.speak_text = tts.build_tts_script(content)
        
    return True


def render_mermaid(code: str) -> None:
    # Clean up markdown code block wrapper if present
    clean_code = code.strip()
    if clean_code.startswith("```"):
        lines = clean_code.splitlines()
        if len(lines) >= 2:
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines[-1].strip().startswith("```"):
                lines = lines[:-1]
        clean_code = "\n".join(lines).strip()
    
    if clean_code.lower().startswith("mermaid"):
        clean_code = clean_code[len("mermaid"):].strip()

    import base64
    b64_code = base64.b64encode(clean_code.encode("utf-8")).decode("utf-8")

    html = f"""
    <div style="background:rgba(255,255,255,0.95);border:1px solid rgba(0,0,0,0.08);
                border-radius:12px;padding:20px;margin:8px 0;">
      <div id="mermaid-container" class="mermaid" style="display:flex;justify-content:center;">Loading map...</div>
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      try {{
        const b64 = "{b64_code}";
        const decoded = atob(b64);
        const bytes = new Uint8Array(decoded.length);
        for (let i = 0; i < decoded.length; i++) {{
            bytes[i] = decoded.charCodeAt(i);
        }}
        const code = new TextDecoder().decode(bytes);
        
        const container = document.getElementById("mermaid-container");
        container.textContent = code;
        
        mermaid.initialize({{
          startOnLoad: false,
          theme: 'default',
          themeVariables: {{
            background: '#ffffff',
            primaryColor: '#6C63FF',
            primaryTextColor: '#000',
            lineColor: '#06B6D4',
            edgeLabelBackground: '#ffffff',
            nodeBorder: '#4F46E5',
            mainBkg: '#F8FAFC'
          }}
        }});
        await mermaid.run({{
          nodes: [container]
        }});
      }} catch (err) {{
        document.getElementById("mermaid-container").innerHTML = `<div style="color:red;font-weight:bold;">Mermaid Render Error: ` + err.message + `</div>`;
      }}
    </script>"""
    components.html(html, height=380, scrolling=True)


def play_tts_queue() -> None:
    if st.session_state.speak_text and not st.session_state.mute_tts:
        text = st.session_state.speak_text
        st.session_state.speak_text = None
        try:
            buf = tts.generate_audio(text)
            st.audio(buf, format="audio/mp3", autoplay=True)
        except Exception as e:
            st.caption(f"⚠️ TTS error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMON UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-title">🎙️ MentorAI</div>
            <div class="sidebar-logo-sub">Classroom Intelligence System</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.provider == "Gemini":
            if st.session_state.gemini_api_key:
                st.markdown('<div class="api-status-connected"><div class="status-dot"></div> Gemini API: Connected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="api-status-disconnected"><div class="status-dot"></div> Gemini API: Key Missing</div>', unsafe_allow_html=True)
        elif st.session_state.provider == "OpenRouter":
            if st.session_state.openrouter_api_key:
                st.markdown('<div class="api-status-connected"><div class="status-dot"></div> OpenRouter: Connected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="api-status-disconnected"><div class="status-dot"></div> OpenRouter: Key Missing</div>', unsafe_allow_html=True)
        else:
            if st.session_state.ollama_model:
                st.markdown('<div class="api-status-connected"><div class="status-dot"></div> Ollama Model: ' + st.session_state.ollama_model + '</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="api-status-disconnected"><div class="status-dot"></div> Ollama: Not Set</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Context Settings
        st.markdown('<span class="sidebar-section-label">📚 Textbook Upload</span>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload PDF/DOCX/TXT", accept_multiple_files=True, label_visibility="collapsed")
        
        if uploaded_files:
            if st.button("Process Documents", use_container_width=True):
                with st.spinner("Processing documents..."):
                    try:
                        for f in uploaded_files:
                            if f.name not in st.session_state.doc_store.sources:
                                st.session_state.doc_store = doc_processor.process_uploaded_file(
                                    file_bytes=f.read(),
                                    filename=f.name,
                                    existing_store=st.session_state.doc_store
                                )
                        st.success(f"Processed! {st.session_state.doc_store.summary()}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
        if not st.session_state.doc_store.is_empty():
            st.caption(f"✅ {st.session_state.doc_store.summary()}")
            if st.button("Clear Textbooks", key="clr_doc"):
                st.session_state.doc_store = doc_processor.DocumentStore()
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        st.markdown('<span class="sidebar-section-label">🎓 Class Settings</span>', unsafe_allow_html=True)
        st.session_state.class_level = st.selectbox(
            "Class Level",
            options=[f"Class {i}" for i in range(1, 13)],
            index=5, # Default Class 6
            help="AI complexity aur vocabulary iske hisaab se adjust hogi."
        )
        st.session_state.subject = st.text_input("Subject (Optional)", value=st.session_state.subject, placeholder="e.g. Science")
        st.session_state.chapter = st.text_input("Chapter (Optional)", value=st.session_state.chapter, placeholder="e.g. Life Processes")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        st.session_state.mute_tts = st.checkbox("🔇 TTS Band Karo", value=st.session_state.mute_tts)

        with st.expander("⚙️ AI Model Configuration", expanded=True):
            options = ["Google Gemini (Cloud - Fast)", "OpenRouter (Cloud - Open-Source)", "Ollama (Local - Offline)"]
            if st.session_state.provider == "Gemini":
                index_val = 0
            elif st.session_state.provider == "OpenRouter":
                index_val = 1
            else:
                index_val = 2
                
            prov = st.selectbox(
                "Provider",
                options=options,
                index=index_val
            )
            if "Gemini" in prov:
                st.session_state.provider = "Gemini"
            elif "OpenRouter" in prov:
                st.session_state.provider = "OpenRouter"
            else:
                st.session_state.provider = "Ollama"
            
            if st.session_state.provider == "Gemini":
                _ak = st.text_input("Gemini API Key", value=st.session_state.gemini_api_key, type="password", placeholder="AIzaSy...")
                if _ak != st.session_state.gemini_api_key:
                    st.session_state.gemini_api_key = _ak
            elif st.session_state.provider == "OpenRouter":
                _ork = st.text_input("OpenRouter API Key", value=st.session_state.openrouter_api_key, type="password", placeholder="sk-or-...")
                if _ork != st.session_state.openrouter_api_key:
                    st.session_state.openrouter_api_key = _ork
                _orm = st.text_input("OpenRouter Model", value=st.session_state.openrouter_model, placeholder="meta-llama/llama-3-8b-instruct:free")
                if _orm != st.session_state.openrouter_model:
                    st.session_state.openrouter_model = _orm
            else:
                _nk = st.text_input("Local Ollama Model", value=st.session_state.ollama_model)
                if _nk != st.session_state.ollama_model:
                    st.session_state.ollama_model = _nk

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        with st.expander("🛠️ Developer Git Sync", expanded=False):
            commit_msg = st.text_input("Commit Message", value="Fix OpenRouter model config and connection reporting")
            if st.button("Commit & Push to GitHub", use_container_width=True):
                with st.spinner("Pushing to GitHub..."):
                    import subprocess
                    try:
                        # Stage all changes
                        subprocess.run(["git", "add", "-A"], capture_output=True, text=True, check=True)
                        # Commit the changes
                        subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
                        # Push to repository
                        res = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True, check=True)
                        st.success("Successfully pushed to GitHub! Reload the deployed app in a minute.")
                        st.text(res.stdout + "\n" + res.stderr)
                    except Exception as err:
                        st.error(f"Git Push Failed: {err}")
                        if "res" in locals():
                            st.text(res.stdout + "\n" + res.stderr)


def render_header():
    st.markdown("""
    <div class="app-header">
        <div class="app-header-badge">✦ Class-Adaptive &nbsp;·&nbsp; RAG Powered &nbsp;·&nbsp; Hinglish</div>
        <div class="gradient-title">MentorAI</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.content:
        _c = st.session_state.content
        _ii = {"CONCEPT": "📖", "QUIZ": "📝", "STORY": "🎭"}.get(_c.intent, "📖")
        
        chunks = getattr(_c, "_rag_chunks", 0)
        if chunks > 0:
            context_status = f"🟢 Grounded ({chunks} chunks)"
        elif not st.session_state.doc_store.is_empty():
            context_status = "🟡 No Textbook Match"
        else:
            context_status = "⚪ General Knowledge"
            
        st.markdown(f"""
        <div class="status-bar animate-in">
            <div class="status-bar-item"><span class="status-bar-label">Topic</span><span class="status-bar-value">{_c.topic}</span></div>
            <div class="status-bar-divider"></div>
            <div class="status-bar-item"><span class="status-bar-label">Audience</span><span class="status-bar-value" style="color:#06B6D4;">🎓 {_c.class_level}</span></div>
            <div class="status-bar-divider"></div>
            <div class="status-bar-item"><span class="status-bar-label">Content Type</span><span class="status-bar-value" style="color:#F43F5E;">{_ii} {_c.intent}</span></div>
            <div class="status-bar-divider"></div>
            <div class="status-bar-item"><span class="status-bar-label">Source</span><span class="status-bar-value" style="color:#10B981;font-size:0.8rem;">{context_status}</span></div>
        </div>
        """, unsafe_allow_html=True)

    play_tts_queue()


def _render_supporting_materials(c, compact: bool = False):
    if c.key_points:
        rows = "".join(
            f'<div class="key-point-row"><div class="kp-bullet">✦</div><div>{p}</div></div>'
            for p in c.key_points
        )
        st.markdown(f"""
        <div class="glass-card animate-in">
            <div class="section-title">📌 Key Points</div>
            {rows}
        </div>
        """, unsafe_allow_html=True)

    if c.visual_summary:
        if compact:
            with st.expander("🗺️ Concept Map dekhein"):
                render_mermaid(c.visual_summary)
        else:
            st.markdown('<div class="section-title">🗺️ Visual Concept Map</div>', unsafe_allow_html=True)
            render_mermaid(c.visual_summary)

    if c.confusing_terms:
        term_rows = "".join(
            f'<div class="term-row"><span class="term-pill">{t.term}</span><span class="term-meaning">{t.simple_meaning}</span></div>'
            for t in c.confusing_terms
        )
        st.markdown(f"""
        <div class="glass-card animate-in" style="background:rgba(244,63,94,0.02)!important;border-color:rgba(244,63,94,0.12)!important;">
            <div class="section-title">🔍 Confusion Detector</div>
            {term_rows}
        </div>
        """, unsafe_allow_html=True)

    # Show understanding check horizontally on the large smart board view
    if c.understanding_check and not compact:
         st.markdown('<div class="section-title">⏱️ Understanding Check</div>', unsafe_allow_html=True)
         cols = st.columns(3)
         colors = ["#10B981", "#F59E0B", "#F43F5E"]
         for i, (col, uq, color) in enumerate(zip(cols, c.understanding_check, colors)):
             with col:
                 st.markdown(f"""
                 <div style="background:var(--bg-card);border:1px solid {color}22;border-top:3px solid {color};border-radius:8px;padding:14px;box-shadow:var(--shadow-card);height:100%;">
                    <div style="font-size:0.75rem;font-weight:700;color:{color};text-transform:uppercase;margin-bottom:6px;">{uq.level}</div>
                    <div style="font-size:0.92rem;color:var(--text-primary);line-height:1.4;margin-bottom:8px;">{uq.question}</div>
                 </div>
                 """, unsafe_allow_html=True)


def render_student_content(compact: bool = False) -> None:
    c = st.session_state.content

    if not c:
        st.markdown("""
        <div class="welcome-splash animate-in">
            <span class="welcome-icon">🎙️</span>
            <div class="welcome-title">Classroom Ready Hai!</div>
            <div class="welcome-desc">
                Teacher Command Center se topic set karein aur voice command dein.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Determine mode from the authoritative session state variable (set by process_command)
    _mode = st.session_state.get("content_mode", "NORMAL")
    is_debate   = (_mode == "DEBATE")
    is_socratic = (_mode == "SOCRATIC")

    # ── CONCEPT / STORY ───────────────────────────────────────────────────
    if c.intent in ("CONCEPT", "STORY"):
        body    = c.story if c.intent == "STORY" else c.explanation

        # Heading adjustments
        if is_socratic:
            heading = f"🤔 Socratic Inquiry: {c.topic}"
        elif is_debate:
            heading = f"🆚 Debate Challenge: {c.topic}"
        else:
            heading = f"{'🎭 Story: ' if c.intent == 'STORY' else ''}{c.topic}"
            
        text_cls = "story-text" if c.intent == "STORY" else "explanation-text"

        st.markdown(f"""
        <div class="glass-card animate-in">
            <div class="topic-heading">{heading}</div>
            <div class="{text_cls}">{body}</div>
        </div>
        """, unsafe_allow_html=True)

        # RENDER INTERACTIVE BOARDS
        if is_socratic:
            st.markdown("""
            <div class="section-title" style="margin-top:24px;">💬 Charcha Board (Socratic Dialogue)</div>
            """, unsafe_allow_html=True)

            # Container for the Socratic chat dialog
            st.markdown('<div class="glass-card animate-in" style="border-left: 5px solid var(--accent-violet) !important; padding: 20px !important;">', unsafe_allow_html=True)

            # Render dialogue bubbles
            chat_history = st.session_state.get("socratic_chat", [])
            for msg in chat_history:
                if msg["role"] == "assistant":
                    st.markdown(f"""
                    <div style="background:rgba(79,70,229,0.04); border:1px solid rgba(79,70,229,0.12); border-left:4px solid var(--primary); border-radius:12px; padding:16px; margin-bottom:12px;">
                        <div style="font-size:0.75rem; font-weight:700; color:var(--primary); text-transform:uppercase; margin-bottom:5px; display:flex; align-items:center; gap:6px;">
                            <span>🎙️</span> <span>MentorAI</span>
                        </div>
                        <div style="font-size:1.02rem; color:var(--text-primary); line-height:1.6;">{msg['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:rgba(8,145,178,0.04); border:1px solid rgba(8,145,178,0.12); border-right:4px solid var(--accent-teal); border-radius:12px; padding:16px; margin-bottom:12px; text-align:right;">
                        <div style="font-size:0.75rem; font-weight:700; color:var(--accent-teal); text-transform:uppercase; margin-bottom:5px; display:flex; align-items:center; justify-content:flex-end; gap:6px;">
                            <span>🎓</span> <span>Student (Class)</span>
                        </div>
                        <div style="font-size:1.02rem; color:var(--text-primary); text-align:left; line-height:1.6; display:inline-block; width:100%;">{msg['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Processing user reply
            student_text = ""
            import hashlib
            import tempfile

            # 1. Text input form
            with st.form(key="socratic_reply_form", clear_on_submit=True):
                col_in, col_btn = st.columns([5, 1.2])
                with col_in:
                    student_text_typed = st.text_input(
                        "socratic_typed_reply",
                        placeholder="Apna jawab yahan likhein (in Hinglish)...",
                        label_visibility="collapsed",
                        key="socratic_typed_input"
                    )
                with col_btn:
                    submitted = st.form_submit_button("🚀 Jawab Bhejein", use_container_width=True)
                
                if submitted and student_text_typed.strip():
                    student_text = student_text_typed.strip()

            # 2. Audio input (Voice)
            st.markdown('<div style="font-size: 0.78rem; font-weight:600; color: var(--text-muted); margin-bottom: 6px; margin-top: 14px;">🎙️ Bolkar jawab dein (Voice Answer):</div>', unsafe_allow_html=True)
            socratic_audio = st.audio_input("socratic_voice_input", label_visibility="collapsed", key="socratic_voice_audio")
            
            if socratic_audio is not None:
                sab = socratic_audio.read()
                sahash = hashlib.md5(sab).hexdigest()
                if st.session_state.get("last_socratic_audio_hash") != sahash:
                    st.session_state.last_socratic_audio_hash = sahash
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(sab)
                        tmp_path = tmp.name
                    try:
                        with st.spinner("🎙️ Transcribing student voice…"):
                            student_text = stt.transcribe_audio(tmp_path)
                    except Exception as e:
                        st.error(f"Voice Transcription Error: {e}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

            if student_text:
                st.session_state.socratic_chat.append({"role": "user", "text": student_text})
                if st.session_state.provider == "Gemini":
                    model_name = "gemini-2.5-flash"
                    api_key = st.session_state.gemini_api_key
                elif st.session_state.provider == "OpenRouter":
                    model_name = st.session_state.openrouter_model
                    api_key = st.session_state.openrouter_api_key
                else:
                    model_name = st.session_state.ollama_model
                    api_key = ""

                context_str = ""
                if not st.session_state.doc_store.is_empty():
                    retrieved = retriever.search(
                        query=student_text,
                        doc_store=st.session_state.doc_store,
                        topic=c.topic,
                        chapter=st.session_state.chapter,
                        subject=st.session_state.subject,
                        top_k=2
                    )
                    context_str = retriever.build_context_string(retrieved)

                with st.spinner("🧠 MentorAI soch raha hai..."):
                    try:
                        res = llm.generate_socratic_response(
                            provider=st.session_state.provider,
                            model_name=model_name,
                            api_key=api_key,
                            topic=c.topic,
                            class_level=st.session_state.class_level,
                            context=context_str,
                            initial_question=c.socratic_question,
                            chat_history=st.session_state.socratic_chat[:-1],
                            student_reply=student_text
                        )
                        st.session_state.socratic_chat.append({
                            "role": "assistant",
                            "text": f"{res['guidance']}\n\n**Sawaal:** {res['next_question']}"
                        })
                        if not st.session_state.mute_tts:
                            st.session_state.speak_text = f"{res['guidance']}. {res['next_question']}"
                    except Exception as exc:
                        st.error(f"MentorAI Socratic response error: {exc}")
                
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        elif is_debate:
            st.markdown("""
            <div class="section-title" style="margin-top:24px;">🆚 Debate Board (Interactive Debate)</div>
            """, unsafe_allow_html=True)

            # Container for the Debate dialogue
            st.markdown('<div class="glass-card animate-in" style="border-left: 5px solid var(--accent-rose) !important; padding: 20px !important;">', unsafe_allow_html=True)

            # Render dialogue bubbles
            debate_history = st.session_state.get("debate_chat", [])
            for msg in debate_history:
                if msg["role"] == "assistant":
                    st.markdown(f"""
                    <div style="background:rgba(225,29,72,0.04); border:1px solid rgba(225,29,72,0.12); border-left:4px solid var(--accent-rose); border-radius:12px; padding:16px; margin-bottom:12px;">
                        <div style="font-size:0.75rem; font-weight:700; color:var(--accent-rose); text-transform:uppercase; margin-bottom:5px; display:flex; align-items:center; gap:6px;">
                            <span>🎙️</span> <span>MentorAI (Opponent)</span>
                        </div>
                        <div style="font-size:1.02rem; color:var(--text-primary); line-height:1.6;">{msg['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:rgba(8,145,178,0.04); border:1px solid rgba(8,145,178,0.12); border-right:4px solid var(--accent-teal); border-radius:12px; padding:16px; margin-bottom:12px; text-align:right;">
                        <div style="font-size:0.75rem; font-weight:700; color:var(--accent-teal); text-transform:uppercase; margin-bottom:5px; display:flex; align-items:center; justify-content:flex-end; gap:6px;">
                            <span>🎓</span> <span>Student (Debater)</span>
                        </div>
                        <div style="font-size:1.02rem; color:var(--text-primary); text-align:left; line-height:1.6; display:inline-block; width:100%;">{msg['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Processing user reply
            debate_text = ""
            import hashlib
            import tempfile

            # 1. Text input form
            with st.form(key="debate_reply_form", clear_on_submit=True):
                col_in, col_btn = st.columns([5, 1.2])
                with col_in:
                    debate_text_typed = st.text_input(
                        "debate_typed_reply",
                        placeholder="Apna argument yahan likhein (in Hinglish)...",
                        label_visibility="collapsed",
                        key="debate_typed_input"
                    )
                with col_btn:
                    submitted = st.form_submit_button("🚀 Submit Tark", use_container_width=True)
                
                if submitted and debate_text_typed.strip():
                    debate_text = debate_text_typed.strip()

            # 2. Audio input (Voice)
            st.markdown('<div style="font-size: 0.78rem; font-weight:600; color: var(--text-muted); margin-bottom: 6px; margin-top: 14px;">🎙️ Bolkar argument dein (Voice Debate):</div>', unsafe_allow_html=True)
            debate_audio = st.audio_input("debate_voice_input", label_visibility="collapsed", key="debate_voice_audio")
            
            if debate_audio is not None:
                sab = debate_audio.read()
                sahash = hashlib.md5(sab).hexdigest()
                if st.session_state.get("last_debate_audio_hash") != sahash:
                    st.session_state.last_debate_audio_hash = sahash
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(sab)
                        tmp_path = tmp.name
                    try:
                        with st.spinner("🎙️ Transcribing student argument…"):
                            debate_text = stt.transcribe_audio(tmp_path)
                    except Exception as e:
                        st.error(f"Voice Transcription Error: {e}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

            if debate_text:
                st.session_state.debate_chat.append({"role": "user", "text": debate_text})
                if st.session_state.provider == "Gemini":
                    model_name = "gemini-2.5-flash"
                    api_key = st.session_state.gemini_api_key
                elif st.session_state.provider == "OpenRouter":
                    model_name = st.session_state.openrouter_model
                    api_key = st.session_state.openrouter_api_key
                else:
                    model_name = st.session_state.ollama_model
                    api_key = ""

                context_str = ""
                if not st.session_state.doc_store.is_empty():
                    retrieved = retriever.search(
                        query=debate_text,
                        doc_store=st.session_state.doc_store,
                        topic=c.topic,
                        chapter=st.session_state.chapter,
                        subject=st.session_state.subject,
                        top_k=2
                    )
                    context_str = retriever.build_context_string(retrieved)

                with st.spinner("🧠 MentorAI counter-argument soch raha hai..."):
                    try:
                        res = llm.generate_debate_response(
                            provider=st.session_state.provider,
                            model_name=model_name,
                            api_key=api_key,
                            topic=c.topic,
                            class_level=st.session_state.class_level,
                            context=context_str,
                            debate_intro=c.explanation,
                            chat_history=st.session_state.debate_chat[:-1],
                            student_reply=debate_text
                        )
                        st.session_state.debate_chat.append({
                            "role": "assistant",
                            "text": f"{res['counter_argument']}\n\n**Challenge:** {res['next_challenge']}"
                        })
                        if not st.session_state.mute_tts:
                            st.session_state.speak_text = f"{res['counter_argument']}. {res['next_challenge']}"
                    except Exception as exc:
                        st.error(f"MentorAI Debate response error: {exc}")
                
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        # Render supporting key points, concept maps, etc.
        # If in Socratic or Debate mode, we hide/wrap them inside an expander so they don't spoil the answers!
        if is_socratic or is_debate:
            with st.expander("📚 Reference Notes & Visual Concept Map (Charcha ke baad reveal karein)"):
                _render_supporting_materials(c, compact)
        else:
            _render_supporting_materials(c, compact)

    # ── QUIZ ─────────────────────────────────────────────────────────────
    elif c.intent == "QUIZ":
        n   = len(c.quiz or [])
        rev = st.session_state.revealed

        st.markdown(f"""
        <div class="quiz-header-card animate-in">
            <div class="quiz-icon">📝</div>
            <div>
                <div class="quiz-topic-title">Quiz: {c.topic}</div>
                <div class="quiz-meta">{n} sawaal &nbsp;·&nbsp; {'✅ Jawab dikhaye ja rahe hain' if rev else '🔒 Jawab teacher reveal karenge'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        letters = ["A", "B", "C", "D"]

        for i, q in enumerate(c.quiz or []):
            opts_html = ""
            for j, opt in enumerate(q.options):
                is_correct = opt.strip() == (q.answer or "").strip()
                letter = letters[j] if j < len(letters) else str(j + 1)

                if rev:
                    cls    = "correct" if is_correct else "incorrect"
                    marker = "✅" if is_correct else "✗"
                else:
                    cls    = ""
                    marker = letter

                opts_html += f'<div class="quiz-option {cls}"><span style="opacity:0.75;font-size:0.82rem;min-width:22px;font-weight:700;">{marker}</span>{opt}</div>'

            exp_html = ""
            if rev and q.explanation:
                exp_html = f'<div class="quiz-explanation-box">💡 {q.explanation}</div>'

            diff_color = {"Easy": "#10B981", "Medium": "#F59E0B", "Hard": "#F43F5E"}.get(q.difficulty, "#6C63FF")

            st.markdown(f"""
            <div class="quiz-card animate-in">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <div class="quiz-q-number">Sawaal {i + 1} / {n}</div>
                    <div style="font-size:0.6rem;font-weight:700;color:{diff_color};border:1px solid {diff_color}44;padding:2px 8px;border-radius:12px;">{q.difficulty}</div>
                </div>
                <div class="quiz-question">{q.question}</div>
                {opts_html}
                {exp_html}
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

def page_teacher_command_center():
    render_header()
    render_sidebar()

    st.markdown('<div class="teacher-pane-header" style="border-radius: var(--radius-lg); margin-bottom: 20px;"><span style="font-size:1.1rem;">🎙️</span><span class="teacher-pane-header-title">Teacher Command Center</span></div>', unsafe_allow_html=True)

    # ── Dashboard Top Row ────────────────────────────────────────────────
    col_doc, col_hist = st.columns([1, 1])
    
    with col_doc:
        st.markdown('<div class="section-title">📚 Document Library</div>', unsafe_allow_html=True)
        if not st.session_state.doc_store.is_empty():
            st.success(f"✅ **Loaded:** {', '.join(st.session_state.doc_store.sources)}")
            st.caption(f"Pages/Chunks extracted: {len(st.session_state.doc_store.chunks)}")
        else:
            st.info("No textbook uploaded yet. Use the sidebar to upload a PDF.")

    with col_hist:
        st.markdown('<div class="section-title">⏱️ Session History</div>', unsafe_allow_html=True)
        if st.session_state.history:
            recent = st.session_state.history[-3:]  # Show last 3
            for h in reversed(recent):
                st.caption(f"• {h['command']}")
        else:
            st.caption("No content generated yet.")

    st.markdown("---")

    st.markdown('<div class="section-title">🎤 Voice Command</div>', unsafe_allow_html=True)
    audio_file = st.audio_input("Voice record", label_visibility="collapsed")

    if audio_file is not None:
        ab    = audio_file.read()
        ahash = hashlib.md5(ab).hexdigest()
        if st.session_state.last_audio_hash != ahash:
            st.session_state.last_audio_hash = ahash
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(ab)
                tmp_path = tmp.name
            try:
                with st.spinner("🎙️ Transcribing locally…"):
                    tr = stt.transcribe_audio(tmp_path)
                st.session_state.transcript = tr
                if tr:
                    if process_command(tr):
                        st.switch_page(page_student)
                else:
                    st.warning("Koi awaaz nahi sunai di. Dobara try karein.")
            except Exception as e:
                st.error(f"Transcription Error: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ── Topic Input ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:18px;">📝 Topic</div>', unsafe_allow_html=True)
    _default_topic = st.session_state.get("topic_input_value", "")
    if not _default_topic and st.session_state.chapter.strip():
        _default_topic = st.session_state.chapter.strip()

    _topic_input = st.text_input(
        "topic_input",
        value=_default_topic,
        placeholder="e.g. Components of Food, Photosynthesis, Gravity...",
        label_visibility="collapsed",
        key="topic_box",
    )
    _topic = _topic_input.strip()
    if _topic != st.session_state.get("topic_input_value", ""):
        st.session_state.topic_input_value = _topic

    if not _topic:
        st.caption("⬆️ Pehle topic likhein, phir neeche button dabayein.")

    # ── Quick Action Buttons ─────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:12px;">⚡ Quick Actions</div>', unsafe_allow_html=True)
    _p1, _p2, _p3 = st.columns(3)
    _actions_1 = [
        (_p1, "📖", "Explain Concept", f"Explain {_topic}"),
        (_p2, "📝", "Generate Quiz",   f"Create a quiz on {_topic}"),
        (_p3, "🎭", "Story Mode",      f"Explain {_topic} as a story"),
    ]
    for _i, (_col, _ico, _lbl, _cmd) in enumerate(_actions_1):
        if _col.button(f"{_ico} {_lbl}", key=f"pre_{_i}", use_container_width=True, disabled=not _topic):
            st.session_state.transcript = _cmd
            if process_command(_cmd):
                st.switch_page(page_student)
                
    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
    _p4, _p5, _p6 = st.columns(3)
    _actions_2 = [
        (_p4, "🤔", "Socratic Question", f"Ask a socratic question about {_topic} to make students think"),
        (_p5, "🌍", "Real-world Example", f"Give a real world example of {_topic}"),
        (_p6, "🆚", "Debate Mode", f"Create a debate topic around {_topic}"),
    ]
    for _i, (_col, _ico, _lbl, _cmd) in enumerate(_actions_2):
        if _col.button(f"{_ico} {_lbl}", key=f"pre2_{_i}", use_container_width=True, disabled=not _topic):
            st.session_state.transcript = _cmd
            if process_command(_cmd):
                st.switch_page(page_student)

    # ── Free-form Command ────────────────────────────────────────────────
    with st.expander("💬 Free-form Command (advanced)", expanded=False):
        with st.form("cmd_form", clear_on_submit=True):
            manual = st.text_input("cmd", placeholder='"Explain the digestive system" ya koi bhi command...', label_visibility="collapsed")
            if st.form_submit_button("🚀 Generate", use_container_width=True):
                if manual:
                    st.session_state.transcript = manual
                    if process_command(manual):
                        st.switch_page(page_student)

    if st.session_state.transcript:
        st.markdown(f"""
        <div class="command-card animate-slide">
            <div class="command-label">🔴 Last Command</div>
            <div class="command-text">"{st.session_state.transcript}"</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.content:
        _cc = st.session_state.content
        st.markdown('<div class="section-title" style="margin-top:18px;">🛠️ Active Controls</div>', unsafe_allow_html=True)

        if _cc.intent == "QUIZ":
            if st.session_state.revealed:
                if st.button("🙈 Hide Answers", use_container_width=True):
                    st.session_state.revealed = False
                    st.rerun()
            else:
                if st.button("👁️ Reveal Answers", use_container_width=True):
                    st.session_state.revealed = True
                    st.rerun()

        if st.button("🔊 Read Aloud", use_container_width=True):
            st.session_state.speak_text = tts.build_tts_script(_cc)
            st.rerun()
        
        # Show understanding check only to teacher
        if _cc.understanding_check:
            st.markdown('<div class="section-title" style="margin-top:18px;">⏱️ Verbal Understanding Check (Teacher Only)</div>', unsafe_allow_html=True)
            for uq in _cc.understanding_check:
                with st.expander(f"{uq.level}: {uq.question}"):
                     st.caption(f"**Expected Hint:** {uq.expected_answer_hint}")




def page_student_learning_board():
    render_header()
    
    st.markdown('<div class="smartboard-mode">', unsafe_allow_html=True)
    if st.session_state.content:
        _cb = st.session_state.content
        _tb1, _tb2, _tb3, _ = st.columns([1.2, 1.2, 1.2, 6])
        with _tb1:
            if _cb.intent == "QUIZ":
                if st.session_state.revealed:
                    if st.button("🙈 Hide Answers", key="b_hide", use_container_width=True):
                        st.session_state.revealed = False
                        st.rerun()
                else:
                    if st.button("👁️ Show Answers", key="b_reveal", use_container_width=True):
                        st.session_state.revealed = True
                        st.rerun()
        with _tb2:
            if st.button("🔊 Read Aloud", key="b_tts", use_container_width=True):
                st.session_state.speak_text = tts.build_tts_script(_cb)
                st.rerun()
        with _tb3:
            if st.button("🗑️ Clear Board", key="b_clear", use_container_width=True):
                st.session_state.content   = None
                st.session_state.revealed  = False
                st.session_state.transcript = ""
                st.session_state.socratic_chat = []
                st.session_state.last_socratic_audio_hash = None
                st.session_state.debate_chat = []
                st.session_state.last_debate_audio_hash = None
                st.rerun()
        st.markdown('<hr style="margin:12px 0;border:0;border-top:1px solid rgba(0,0,0,0.08);">', unsafe_allow_html=True)

    render_student_content(compact=False)
    st.markdown('</div>', unsafe_allow_html=True)


def page_session_log():
    render_header()
    render_sidebar()

    st.markdown('<div class="section-title">📋 Session Log</div>', unsafe_allow_html=True)

    if not st.session_state.history:
         st.info("No commands run yet.")
    else:
        for _i, _h in enumerate(reversed(st.session_state.history)):
            _idx = len(st.session_state.history) - _i
            _int = _h["content"].intent
            _top = _h["content"].topic
            
            _h1, _h2 = st.columns([5, 1])
            with _h1:
                st.markdown(f"""
                <div class="history-item">
                    <div class="history-cmd">#{_idx} — {_h['command']}</div>
                    <div class="history-meta" style="margin-top:5px; color: var(--text-secondary);">
                        Topic: <strong style="color: var(--primary);">{_top}</strong> | Type: <strong>{_int}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with _h2:
                if st.button("↩ Reload", key=f"hr_{_i}", use_container_width=True):
                    st.session_state.content   = _h["content"]
                    st.session_state.transcript = _h["command"]
                    st.session_state.revealed  = False
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

page_teacher = st.Page(page_teacher_command_center, title="Teacher Command Center", icon="🎙️")
page_student = st.Page(page_student_learning_board, title="Student Learning Board", icon="📺")
page_log = st.Page(page_session_log, title="Session Log", icon="📋")

pg = st.navigation([page_teacher, page_student, page_log])

pg.run()
