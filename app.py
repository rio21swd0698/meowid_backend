from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import requests

app = Flask(__name__)
CORS(app)

# === KONFIGURASI ===
MODEL_PATH = 'meowid_model.h5'
MODEL_FILE_ID = '1vJc5TZzekDR8EtmHWWC6e3CHxZ1Sv5Mp'  # ← Google Drive file ID
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.4))  # 40%

class_names = ['Anggora', 'Bengal', 'Persian', 'Siamese', 'Sphynx', 'Tabby']

# === DOWNLOAD MODEL JIKA BELUM ADA ===


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Mengunduh model dari Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("✅ Model berhasil diunduh.")
        else:
            print("❌ Gagal mengunduh model:", response.status_code)
            raise Exception("Gagal mengunduh model dari Google Drive")


download_model()

# === LOAD MODEL ===
model = load_model(MODEL_PATH)
print("✅ Model loaded. Input shape:", model.input_shape)

# === ENDPOINT CEK ===


@app.route('/')
def index():
    return jsonify({'message': '🐾 MeowID Backend is running!'})

# === ENDPOINT PREDIKSI ===


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    try:
        # Preprocessing gambar
        img = Image.open(file.stream).convert("RGB").resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)

        # Prediksi
        prediction = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        if confidence >= CONFIDENCE_THRESHOLD:
            return jsonify({
                'class': predicted_class,
                'confidence': f'{confidence * 100:.2f}%',
                'status': 'success'
            })
        else:
            return jsonify({
                'class': None,
                'confidence': f'{confidence * 100:.2f}%',
                'status': 'uncertain',
                'message': 'Model tidak yakin dengan prediksi ini.'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === JALANKAN LOKAL / RAILWAY ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
