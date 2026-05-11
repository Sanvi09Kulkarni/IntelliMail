import streamlit as st
import requests
import pandas as pd


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Email Analyzer",
    page_icon="📧",
    layout="wide"
)


# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.stApp {
    background-color: #030712;
    color: white;
}

textarea {
    font-size: 16px !important;
}

div[data-testid="stMetric"] {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #1F2937;
}

div[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================

st.markdown("""
# 📧 AI Email Intent Analyzer

### Smart transformer-based email understanding system
""")


# =========================
# EMAIL INPUT
# =========================

email_text = st.text_area(
    "Paste Email Content",
    height=250,
    placeholder="Paste your email here..."
)


# =========================
# ANALYZE BUTTON
# =========================

if st.button("Analyze Email"):

    if email_text.strip() == "":
        st.warning("Please enter email text.")

    else:

        # =========================
        # API REQUEST
        # =========================

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": email_text}
        )

        result = response.json()

        intent = result["intent"]
        confidence = result["confidence"]

        # =========================
        # RESULT SECTION
        # =========================

        st.markdown("## 📌 Prediction Result")

        emoji_map = {
            "meeting": "🔵",
            "deadline": "🟠",
            "support": "🔴",
            "finance": "🟢",
            "promotion": "🟣",
            "personal": "⚪",
            "spam": "🚨"
        }

        emoji = emoji_map.get(intent, "📧")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label=f"{emoji} Predicted Intent",
                value=intent.upper()
            )

        with col2:
            st.metric(
                label="🎯 Confidence",
                value=f"{round(confidence * 100, 2)}%"
            )

        # =========================
        # CONFIDENCE BAR
        # =========================

        st.markdown("### 📊 Confidence Level")

        st.progress(float(confidence))

        st.caption(f"{round(confidence * 100, 2)}% confidence")


        # =========================
        # SUMMARY
        # =========================

        summary_map = {
            "meeting": "This email is related to a meeting or discussion.",
            "deadline": "This email contains an important deadline or submission.",
            "support": "This email reports a technical or account-related issue.",
            "finance": "This email is related to financial activity or payments.",
            "promotion": "This email contains promotional or marketing content.",
            "personal": "This email appears to be personal communication.",
            "spam": "This email may contain spam or suspicious content."
        }

        st.markdown("### 🧠 AI Summary")

        st.success(
            summary_map.get(intent, "No summary available.")
        )


# =========================
# EMAIL HISTORY
# =========================

st.markdown("---")
st.markdown("## 🕘 Recent Email History")

history_response = requests.get(
    "http://127.0.0.1:8000/emails"
)

history_data = history_response.json()

if "data" in history_data:

    df = pd.DataFrame(history_data["data"])

    if not df.empty:

        df["confidence"] = (
            df["confidence"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            df[["text", "prediction", "confidence", "timestamp"]],
            use_container_width=True,
            hide_index=True
        )