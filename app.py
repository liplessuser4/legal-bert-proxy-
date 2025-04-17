from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Laad NER pipeline voor Nederlands
nlp = pipeline("ner", model="GroNLP/bert-base-dutch-cased", grouped_entities=True)

@app.route("/legal-bert/ner", methods=["POST"])
def named_entity_recognition():
    data = request.json
    tekst = data.get("tekst")

    if not tekst:
        return jsonify({"error": "Geen tekst ontvangen"}), 400

    try:
        output = nlp(tekst)
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check
@app.route("/", methods=["GET"])
def health():
    return "Legal BERT proxy actief", 200

if __name__ == "__main__":
    app.run(debug=True)
