from flask import Flask, request, jsonify
import threading
import json
import numpy as np
import torch
import threading
import logging
from utils import generate_embedding
# Initialize models globally to load them once

app = Flask(__name__)

@app.route('/.well-known/ready', methods=['GET'])
def readiness_check():
    return "Ready", 200

@app.route('/meta', methods=['GET'])
def readiness_check_2():
    return jsonify({'status': 'Ready'}), 200

@app.route('/vectors', methods=['POST']) 
def vectorize():
    try:
        try:
            data = request.json.get('text')
        except Exception as e:
            try:
                data = request.data.decode("utf-8")
            except Exception as e:
                print(e)
        text = json.loads(data)
        if isinstance(text, str):
            text = [text]
        else:
            text =text['text']
            
        embeddings = generate_embedding(text)

        return jsonify({'vector': embeddings})


    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
app.logger.disabled = True
# Get the Flask app's logger
log = logging.getLogger('werkzeug')
# Set logging level (ERROR or CRITICAL suppresses routing logs)
log.setLevel(logging.ERROR)
def run_app():
    app.run(host='0.0.0.0', port=5001, debug=False)


def _vectorizer_ready(url="http://127.0.0.1:5001/.well-known/ready"):
    try:
        import requests
        return requests.get(url, timeout=1).ok
    except Exception:
        return False


def start_vectorizer(wait_timeout=60):
    """Sobe o vectorizer Flask na 5001 se ainda não estiver rodando e espera ficar ready.
    Idempotente: seguro chamar/importar mais de uma vez."""
    import time
    if not _vectorizer_ready():
        threading.Thread(target=run_app, daemon=True).start()
    for _ in range(wait_timeout):
        if _vectorizer_ready():
            print("Vectorizer ready on http://127.0.0.1:5001")
            return
        time.sleep(1)
    raise RuntimeError("Flask vectorizer não ficou ready na porta 5001")


# Auto-start no import (idempotente)
start_vectorizer()
