import os
if os.path.exists("F:\\hf_cache"):
    os.environ["HF_HOME"] = "F:\\hf_cache"

import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from arabert.preprocess import ArabertPreprocessor

MODEL_NAME = "malmarjeh/mbert2mbert-arabic-text-summarization"

st.set_page_config(page_title="YouTube Video Summarizer", page_icon="🎥")
st.title("🎥 YouTube Video Summarizer")
st.caption("URL → transcript → Arabic summary (mBERT2mBERT)")


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
        if video_id:
            return video_id
    if host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        qs = parse_qs(parsed.query)
        if qs.get("v"):
            return qs["v"][0]
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            return parts[1]
    raise ValueError("No video id found in this URL")


@st.cache_resource(show_spinner="Loading model (first time only)...")
def load_model():
    preprocessor = ArabertPreprocessor(model_name="")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return preprocessor, tokenizer, pipe


def get_transcript(video_id: str) -> str:
    fetched = YouTubeTranscriptApi().fetch(video_id, languages=["ar", "en"])
    return "\n".join(snippet.text for snippet in fetched)


def summarize(text: str, chunk_words: int = 200) -> str:
    preprocessor, tokenizer, pipe = load_model()
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
    summaries = []
    for chunk in chunks:
        clean = preprocessor.preprocess(chunk)
        out = pipe(
            clean,
            truncation=True,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=3,
            repetition_penalty=3.0,
            max_length=200,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )[0]["generated_text"]
        summaries.append(out)
    return "\n\n".join(summaries)


url = st.text_input("Paste a YouTube URL")

if st.button("Summarize") and url:
    try:
        video_id = extract_video_id(url)
        with st.spinner("Fetching transcript..."):
            text = get_transcript(video_id)
        with st.spinner("Summarizing..."):
            summary = summarize(text)
        st.subheader("Summary")
        st.write(summary)
        with st.expander("Full transcript"):
            st.write(text)
    except Exception as e:
        st.error(f"Error: {e}")
