import atexit
import tempfile
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file
from googletrans import Translator
from gtts import gTTS

from core.recognition import GestureRecognizer
from core.translations import TRANSLATIONS


app = Flask(__name__)

camera = None
recognizer = None
translator = Translator()

last_prediction = {
    "prediction": None,
    "label": None,
    "confidence": 0.0,
}

LANGUAGE_CODES = {
    "français": "fr",
    "anglais":  "en",
    "espagnol": "es",
    "chinois":  "zh-CN",
}


def get_camera():
    global camera

    if camera is None:
        camera = cv2.VideoCapture(0)

    return camera


def get_recognizer():
    global recognizer

    if recognizer is None:
        recognizer = GestureRecognizer(enable_voice=True)

    return recognizer


def _translate_label(label, language):
    texte_fr = TRANSLATIONS.get(label.lower(), label)

    if language == "français":
        return texte_fr

    lang_code = LANGUAGE_CODES.get(language)
    if not lang_code:
        return texte_fr

    try:
        return translator.translate(texte_fr, src="fr", dest=lang_code).text
    except Exception as e:
        print(f"[translation error] {e}")
        return texte_fr


def generate_frames(show_landmarks=True, enable_voice=False):
    cap = get_camera()
    engine = get_recognizer()

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        processed, prediction, confidence = engine.process_frame(
            frame,
            show_landmarks=show_landmarks,
            enable_voice=enable_voice,
        )

        if prediction is not None:
            last_prediction["prediction"] = prediction["display_text"]
            last_prediction["label"] = prediction["name"]
            last_prediction["confidence"] = confidence

        success, buf = cv2.imencode(".jpg", processed)

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    show_landmarks = request.args.get("landmarks", "true").lower() == "true"
    enable_voice = request.args.get("voice", "false").lower() == "true"

    return Response(
        generate_frames(show_landmarks, enable_voice),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/prediction")
def prediction():
    return jsonify(last_prediction)


@app.route("/get_translation")
def get_translation():
    label = request.args.get("label", "")
    language = request.args.get("language", "français")

    return jsonify({"text": _translate_label(label, language)})


@app.route("/translate_speak")
def translate_speak():
    label = request.args.get("label", "")
    language = request.args.get("language", "français")

    if not label or language not in LANGUAGE_CODES:
        return jsonify({"error": "Paramètres invalides"}), 400

    try:
        texte = _translate_label(label, language)
        lang_code = LANGUAGE_CODES[language]

        tmp = Path(tempfile.mktemp(suffix=".mp3"))
        gTTS(text=texte, lang=lang_code).save(str(tmp))

        return send_file(str(tmp), mimetype="audio/mpeg", as_attachment=False)

    except Exception as e:
        print(f"[tts error] {e}")
        return jsonify({"error": str(e)}), 500


def cleanup():
    global camera, recognizer

    if camera is not None:
        camera.release()

    if recognizer is not None:
        recognizer.close()


atexit.register(cleanup)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
