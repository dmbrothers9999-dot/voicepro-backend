import os
import logging
from datetime import datetime
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter
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
    """Converts any audio file to 22050Hz mono WAV using FFmpeg"""
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
        audio, sr = sf.read(file_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr


def process_clean_vocals(audio, sr, intensity='medium'):
    """
    Soft Spectral Subtraction Algorithm with Spectral Floor
    Preserves human voice formants while smoothly suppressing background noise.
    """
    n_fft = 2048
    hop_length = 512

    # 1. Short-Time Fourier Transform (STFT)
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    if magnitude.shape[1] == 0:
        return audio

    # 2. Smart Noise Floor Estimation (Find quietest 10% frames)
    frame_energies = np.sum(magnitude ** 2, axis=0)
    quiet_count = max(1, int(0.10 * len(frame_energies)))
    quiet_indices = np.argsort(frame_energies)[:quiet_count]
    noise_profile = np.mean(magnitude[:, quiet_indices], axis=1, keepdims=True)

    # 3. Parameters for preserving vocals
    if intensity == 'light':
        alpha = 1.1  # Subtraction multiplier
        beta = 0.25  # Spectral floor (25% noise floor retained = zero vocal damage)
    elif intensity == 'heavy':
        alpha = 2.0
        beta = 0.12  # 12% floor
    else:  # medium
        alpha = 1.5
        beta = 0.18  # 18% floor (balanced)

    # 4. Soft Subtraction Gain Mask
    subtracted = magnitude - (alpha * noise_profile)
    gain_mask = subtracted / (magnitude + 1e-10)
    gain_mask = np.maximum(gain_mask, beta)  # Prevents vocal cutting

    # 5. Smooth the mask to remove robotic / metallic artifacts
    smoothed_mask = gaussian_filter(gain_mask, sigma=(1.0, 1.5))

    # 6. Reconstruct Audio via Inverse STFT
    cleaned_stft = magnitude * smoothed_mask * np.exp(1j * phase)
    _, cleaned_audio = signal.istft(cleaned_stft, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Match exact audio length
    if len(cleaned_audio) > len(audio):
        cleaned_audio = cleaned_audio[:len(audio)]
    elif len(cleaned_audio) < len(audio):
        cleaned_audio = np.pad(cleaned_audio, (0, len(audio) - len(cleaned_audio)))

    cleaned_audio = np.nan_to_num(cleaned_audio)
    cleaned_audio = np.clip(cleaned_audio, -1.0, 1.0)

    return cleaned_audio


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

        # Load audio safely
        audio, sr = load_audio_ffmpeg(path)

        # Process clean vocals (Vocal Protection Active)
        cleaned = process_clean_vocals(audio, sr, intensity)

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