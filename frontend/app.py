import streamlit as st
import requests
import time

API_URL = "https://quora-duplicate-question-zuq9.onrender.com"

st.set_page_config(page_title="SemantiQ", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── GLOBAL ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #212121;
    color: #ffffff;
}
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 640px !important;
    margin: 0 auto !important;
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
}

/* ── HERO ── */
.hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 2.5rem 1rem 2rem;
}

.hero-title {
    font-size: clamp(2.6rem, 7vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    width: 100%;
    text-align: center !important;
}

.hero-title .cyan { color: #00FFFF; }

.hero-sub {
    font-size: 1rem;
    color: #888888;
    max-width: 420px;
    margin: 1rem auto 0;
    text-align: center;
}

/* ── INPUTS ── */
div[data-testid="stTextInput"] input {
    text-align: left !important;
    padding-left: 1rem !important;
}

/* ── BUTTON ── */
div[data-testid="stButton"] {
    display: flex;
    justify-content: flex-start;
}

/* ── ANALYZING ANIMATION ── */
.analyzing-box {
    margin-top: 1.5rem;
    padding: 1.2rem;
    border-radius: 12px;
    background: #2a2a2a;
    border: 1px solid #333;
    text-align: center;
}

.loader {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-bottom: 10px;
}

.dot {
    width: 8px;
    height: 8px;
    background: #00FFFF;
    border-radius: 50%;
    animation: bounce 1.2s infinite ease-in-out;
}

.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
    40% { transform: scale(1); opacity: 1; }
}

.progress-bar {
    height: 6px;
    background: #333;
    border-radius: 50px;
    overflow: hidden;
    margin-top: 10px;
}

.progress-fill {
    height: 100%;
    width: 100%;
    background: linear-gradient(90deg, #008080, #00FFFF);
    animation: loading 1.5s infinite;
}

@keyframes loading {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* ── RESULT ── */
.result-box {
    margin-top: 1.5rem;
    border-radius: 14px;
    border: 1px solid;
    text-align: center;
}
.result-box.dup { border-color: #ff6b35; }
.result-box.nodup { border-color: #00FFFF; }

.result-head {
    padding: 1.2rem;
    font-size: 1.3rem;
}

.result-foot {
    padding: 1rem;
    font-size: 0.9rem;
}

/* ── FOOTER ── */
.footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.75rem;
    color: #444;
}
.footer span { color: #008080; }

</style>
""", unsafe_allow_html=True)

# hero
st.markdown("""
<div class="hero">
    <h1 class="hero-title">
        Are your questions<br>
        <span class="cyan">saying the same thing?</span>
    </h1>
    <p class="hero-sub">
        Enter any two questions and our ML model instantly detects
        whether they're semantically duplicate — with confidence scoring.
    </p>
</div>
""", unsafe_allow_html=True)

# input
q1 = st.text_input("First Question")
q2 = st.text_input("Second Question")

def fetch_prediction(q1, q2):
    try:
        res = requests.post(
            f"{API_URL}/predict",
            json={"question1": q1, "question2": q2},
            timeout=20
        )
        if res.status_code == 200:
            data = res.json()
            return data.get("prediction"), data.get("confidence")
    except Exception as e:
        st.error(e)
    return None, None

# button
if st.button("Analyse Questions →"):
    if not q1 or not q2:
        st.warning("Enter both questions")
    else:
        placeholder = st.empty()

        # SHOW ANIMATION
        placeholder.markdown("""
        <div class="analyzing-box">
            <div class="loader">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
            <div>Analyzing your questions...</div>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # simulate small delay
        time.sleep(1)

        prediction, confidence = fetch_prediction(q1, q2)

        # result
        if prediction:
            is_dup = prediction.lower() == "duplicate"
            cls = "dup" if is_dup else "nodup"

            placeholder.markdown(f"""
            <div class="result-box {cls}">
                <div class="result-head">
                    {prediction}
                </div>
                <div class="result-foot">
                    Confidence: {round((confidence or 0)*100,1)}%
                </div>
            </div>
            """, unsafe_allow_html=True)

# footer
st.markdown("""
<div class="footer">
<span>SemantiQ</span> · Semantic Similarity Engine · ML Powered
</div>
""", unsafe_allow_html=True)