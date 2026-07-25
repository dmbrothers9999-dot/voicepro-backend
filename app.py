import os
import logging
from datetime import datetime
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import soundfile as sf
from scipy import signal
import ffmpeg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def load_audio_ffmpeg(file_path):
    """Converts any audio format to standard WAV using FFmpeg seamlessly"""
    temp_wav = file_path + "_temp.wav"
    try:
        (
            ffmpeg
            .input(file_path)
            .output(temp_wav, ar=22050, ac=1, acodec='pcm_s16le')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio, sr = sf.read(temp_wav, dtype='float32')
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return audio, sr
    except Exception as e:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        # Direct fallback read
        audio, sr = sf.read(file_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr


@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    return jsonify({'status': 'VoicePro Server Online', 'developer': 'Daini Magician'})


@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        clean_filename = file.filename.replace(" ", "_")
        filepath = os.path.join(UPLOAD_FOLDER, clean_filename)
        file.save(filepath)

        return jsonify({'success': True, 'filename': clean_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reduce-noise', methods=['POST', 'OPTIONS'])
def handle_noise_reduction():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        data = request.json or {}
        filename = data.get('filename')
        intensity = data.get('intensity', 'medium')

        if not filename:
            return jsonify({'error': 'Filename missing!'}), 400

        path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(path):
            path = os.path.join(PROCESSED_FOLDER, filename)

        if not os.path.exists(path):
            return jsonify({'error': 'File not found. Please upload again.'}), 404

        # Load Audio via FFmpeg (No Librosa / No CFFI errors)
        audio, sr = load_audio_ffmpeg(path)

        # Highpass Filter + Noise Gate (0.2s Execution Time)
        cutoff = 300 if intensity == 'light' else (500 if intensity == 'medium' else 800)
        b, a = signal.butter(4, cutoff / (sr / 2), btype='highpass')
        filtered = signal.filtfilt(b, a, audio)

        thresh = 0.01 if intensity == 'light' else (0.02 if intensity == 'medium' else 0.04)
        mask = np.abs(filtered) > thresh
        
        # Smooth mask window to prevent audio clicks
        kernel_size = int(sr * 0.01)
        if kernel_size > 0:
            smooth_mask = np.convolve(mask.astype(float), np.ones(kernel_size) / kernel_size, mode='same')
        else:
            smooth_mask = mask.astype(float)

        cleaned = filtered * smooth_mask
        cleaned = np.nan_to_num(cleaned)
        cleaned = np.clip(cleaned, -1.0, 1.0)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f"denoised_{timestamp}.wav"
        out_path = os.path.join(PROCESSED_FOLDER, out_name)

        sf.write(out_path, cleaned, sr)
        return jsonify({'success': True, 'denoised_file': out_name})

    except Exception as e:
        logger.error(f"Denoise Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export-audio', methods=['POST', 'OPTIONS'])
def handle_export():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        data = request.json or {}
        filename = data.get('filename')
        fmt = data.get('format', 'mp3').lower()
        bitrate = data.get('bitrate', '192')

        if not filename:
            return jsonify({'error': 'Filename missing'}), 400

        path = os.path.join(PROCESSED_FOLDER, filename)
        if not os.path.exists(path):
            path = os.path.join(UPLOAD_FOLDER, filename)

        if not os.path.exists(path):
            return jsonify({'error': 'File not found'}), 404

        out_name = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
        out_path = os.path.join(PROCESSED_FOLDER, out_name)

        if fmt == 'wav':
            audio, sr = sf.read(path)
            sf.write(out_path, audio, sr)
        else:
            codec_map = {'mp3': 'libmp3lame', 'aac': 'aac', 'ogg': 'libvorbis', 'flac': 'flac'}
            codec = codec_map.get(fmt, 'libmp3lame')

            (
                ffmpeg
                .input(path)
                .output(out_path, acodec=codec, audio_bitrate=f"{bitrate}k")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

        return jsonify({'success': True, 'output_file': out_name})

    except Exception as e:
        logger.error(f"Export Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET', 'OPTIONS'])
def download_file(filename):
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(path):
        path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        path,
        as_attachment=True,
        download_name=filename,
        mimetype='application/octet-stream'
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)