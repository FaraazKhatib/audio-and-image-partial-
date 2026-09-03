import os
import re
import warnings
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from PIL import Image
import io
import base64

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ── Model loading ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("Loading fake news models...")
news_model = joblib.load(os.path.join(MODELS_DIR, "fake_news_model.pkl"))
tfidf_vec   = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
print("✅ News models loaded")

print("Loading deepfake image model...")
import tensorflow as tf
image_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "efficientnetv2b2_deepfake_final.hdf5"),
    compile=False
)
print("✅ Image model loaded")

IMG_SIZE = (128, 128)

# ── Text preprocessing (mirrors training pipeline) ────────────────────────────
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem   import WordNetLemmatizer
    _stop  = set(stopwords.words("english"))
    _lemma = WordNetLemmatizer()
    USE_NLTK = True
except Exception:
    USE_NLTK = False
    # Minimal English stopwords fallback
    _stop = {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "yourself","he","him","his","himself","she","her","hers","herself","it",
        "its","itself","they","them","their","theirs","themselves","what","which",
        "who","whom","this","that","these","those","am","is","are","was","were",
        "be","been","being","have","has","had","having","do","does","did","doing",
        "a","an","the","and","but","if","or","because","as","until","while","of",
        "at","by","for","with","about","against","between","into","through",
        "during","before","after","above","below","to","from","up","down","in",
        "out","on","off","over","under","again","further","then","once","here",
        "there","when","where","why","how","all","both","each","few","more","most",
        "other","some","such","no","nor","not","only","own","same","so","than",
        "too","very","s","t","can","will","just","don","should","now","d","ll","m",
        "o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
        "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
        "won","wouldn"
    }


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    if USE_NLTK:
        tokens = [_lemma.lemmatize(t) for t in tokens if t not in _stop and len(t) > 1]
    else:
        tokens = [t for t in tokens if t not in _stop and len(t) > 1]
    return " ".join(tokens)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) < 20:
        return jsonify({"error": "Text too short. Provide at least 20 characters."}), 400

    processed = preprocess_text(text)
    X = tfidf_vec.transform([processed])
    pred  = int(news_model.predict(X)[0])
    proba = news_model.predict_proba(X)[0]

    # Class mapping: 1 = REAL, 0 = FAKE
    is_fake       = pred == 0
    confidence    = float(proba[0]) if is_fake else float(proba[1])
    label         = "FAKE" if is_fake else "REAL"
    fake_score    = float(proba[0])   # probability of being fake
    real_score    = float(proba[1])   # probability of being real

    return jsonify({
        "label":      label,
        "is_fake":    is_fake,
        "confidence": round(confidence * 100, 2),
        "fake_score": round(fake_score * 100, 2),
        "real_score": round(real_score * 100, 2),
        "word_count": len(text.split()),
    })


@app.route("/api/analyze-image", methods=["POST"])
def analyze_image():
    # Accept either multipart/form-data or JSON with base64
    if request.content_type and "multipart/form-data" in request.content_type:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        file = request.files["image"]
        img_bytes = file.read()
    else:
        data = request.get_json(force=True)
        b64 = data.get("image", "")
        if not b64:
            return jsonify({"error": "No image data provided"}), 400
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)

    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image format"}), 400

    pil_img   = pil_img.resize(IMG_SIZE)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = image_model.predict(img_array, verbose=0)[0]

    # Model output: index 0 = FAKE, index 1 = REAL (softmax)
    fake_score = float(preds[0])
    real_score = float(preds[1])
    is_fake    = fake_score > real_score
    label      = "DEEPFAKE" if is_fake else "REAL"
    confidence = fake_score if is_fake else real_score

    return jsonify({
        "label":      label,
        "is_fake":    is_fake,
        "confidence": round(confidence * 100, 2),
        "fake_score": round(fake_score * 100, 2),
        "real_score": round(real_score * 100, 2),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)