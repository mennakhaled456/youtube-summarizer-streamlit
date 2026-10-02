# 🎥 YouTube Video Summarizer (Streamlit)

A Streamlit web app that takes a YouTube link, extracts the video transcript, and generates an Arabic summary using a Hugging Face model.

## How it works
1. Paste a YouTube URL
2. The transcript is fetched with `youtube-transcript-api` (Arabic or English captions)
3. The text is split into chunks and cleaned with AraBERT preprocessing
4. `malmarjeh/mbert2mbert-arabic-text-summarization` generates the summary

## Tech stack
- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- youtube-transcript-api
- AraBERT

## Run locally
```bash
git clone https://github.com/mennakhaled456/youtube-summarizer-streamlit.git
cd youtube-summarizer-streamlit
pip install -r requirements.txt
streamlit run app.py
```


## Notes
- The video must have captions available.
- Original prototype is in `summary.ipynb`.

Built for Lab 4 of the Generative AI Internship.