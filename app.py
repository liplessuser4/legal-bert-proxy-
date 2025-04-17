from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

app = Flask(__name__)

# 📌 Laad Legal Dutch BERT-model (NER)
MODEL_NAME = "uvacreate/bert-base-dutch-legal"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
except Exception as e:
    print("Fout bij laden van model:", str(e))
    raise

# 🔎 Endpoint: Named Entity Recognition
@app.route("/legal-bert/ner", methods=["POST"])
def named_entity_recognition():
    data = request.json
    tekst = data.get("tekst", "")

    if not tekst:
        return jsonify({"error": "Geen tekst ontvangen"}), 400

    try:
        resultaten = nlp(tekst)
        return jsonify(resultaten)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Health check
@app.route("/", methods=["GET"])
def health_check():
    return "Legal BERT proxy draait!", 200

if __name__ == "__main__":
    app.run(debug=True, port=8080)
