import base64
import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import requests

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        data = request.get_json()
        audio_b64 = data['audio']
        audio_bytes = base64.b64decode(audio_b64)

        with open("temp_audio.webm", "wb") as f:
            f.write(audio_bytes)

        files = {
            "file": ("audio.webm", open("temp_audio.webm", "rb"), "audio/webm")
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data={"model": "whisper-large-v3"}
        )

        if not response.ok:
            return jsonify({"error": "Transcription API failed"}), 500

        transcription_json = response.json()
        transcription_text = transcription_json.get("text")

        if not transcription_text:
            return jsonify({"error": "No transcription text received"}), 500

        return jsonify({"text": transcription_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/summarise", methods=["POST"])
def summarise():
    try:
        data = request.get_json()
        user_text = data['text']

        summary_response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gemma2-9b-it",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Summarize the following transcription into a clear paragraph and bullet points where appropriate."},
                    {"role": "user", "content": user_text}
                ]
            }
        )

        if not summary_response.ok:
            return jsonify({"error": "Summary API failed"}), 500

        summary_json = summary_response.json()
        summary = summary_json.get("choices", [{}])[0].get("message", {}).get("content", "No summary returned")

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
from googletrans import Translator

@app.route("/translate", methods=["POST"])
def translate_text():
    data = request.json
    text = data.get("text", "")
    target_lang = data.get("target_lang", "en")
    
    translator = Translator()
    translated = translator.translate(text, dest=target_lang)
    
    return jsonify({"translated_text": translated.text})


if __name__ == "__main__":
    app.run(debug=True)
