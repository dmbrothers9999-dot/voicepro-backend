import os
from datetime import datetime
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import soundfile as sf
import ffmpeg

app = Flask(__name__)
CORS(app)  # Blogger connection ke liye zaroori hai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return jsonify({'status': 'VoicePro Server Running Live!'})

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return jsonify({'success': True, 'filename': file.filename})

@app.route('/api/reduce-noise', methods=['POST'])
def reduce_noise():
    data = request.json
    filename = data.get('filename')
    intensity = data.get('intensity', 'medium')

    input_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(input_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        audio, sr = librosa.load(input_path, sr=44100)
        D = librosa.stft(audio)
        magnitude, phase = np.abs(D), np.angle(D)
        noise_floor = np.percentile(magnitude, 15)
        
        threshold = 1.5 if intensity == 'light' else (2.0 if intensity == 'medium' else 2.8)
        mask = magnitude > (noise_floor * threshold)
        cleaned_mag = magnitude * mask
        cleaned_audio = librosa.istft(cleaned_mag * np.exp(1j * phase))
        
        out_name = f"denoised_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        out_path = os.path.join(PROCESSED_FOLDER, out_name)
        sf.write(out_path, cleaned_audio, sr)

        return jsonify({'success': True, 'denoised_file': out_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-audio', methods=['POST'])
def export_audio():
    data = request.json
    filename = data.get('filename')
    fmt = data.get('format', 'mp3')
    bitrate = data.get('bitrate', '192')
    
    path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(path):
        path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404

    try:
        audio, sr = librosa.load(path, sr=None)
        out_name = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
        out_path = os.path.join(PROCESSED_FOLDER, out_name)

        if fmt in ['wav', 'flac']:
            sf.write(out_path, audio, sr)
        else:
            temp_wav = os.path.join(UPLOAD_FOLDER, 'temp.wav')
            sf.write(temp_wav, audio, sr)
            codec_map = {'mp3': 'libmp3lame', 'aac': 'aac', 'ogg': 'libvorbis'}
            stream = ffmpeg.input(temp_wav)
            stream = ffmpeg.output(stream, out_path, acodec=codec_map.get(fmt, 'libmp3lame'), audio_bitrate=f"{bitrate}k")
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        return jsonify({'success': True, 'output_file': out_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download(filename):
    path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(path):
        path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)