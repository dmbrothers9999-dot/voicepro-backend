import os
import sys
import logging
from datetime import datetime
from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter
import ffmpeg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyInstaller Path Resolver
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

template_dir = get_resource_path('templates')
static_dir = get_resource_path('static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.join(os.path.expanduser('~'), 'VoiceProStudio_Uploads')
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def load_audio_ffmpeg(file_path):
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


def remove_background_noise_engine(audio, sr, noise_pct=70):
    """
    Dedicated Background Noise Removal Engine
    Uses Soft Spectral Subtraction to eliminate background hums, fans, and street noise.
    """
    # 1. Highpass Sub-Hum Filter (< 80Hz)
    b_hp, a_hp = signal.butter(4, 80 / (sr / 2), btype='highpass')
    audio = signal.filtfilt(b_hp, a_hp, audio)

    # 2. Soft Spectral Subtraction
    if noise_pct > 0:
        alpha = 1.0 + (noise_pct / 100.0) * 1.6
        beta = max(0.08, 0.28 - (noise_pct / 100.0) * 0.20)

        n_fft, hop_length = 2048, 512
        f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        if magnitude.shape[1] > 0:
            frame_energies = np.sum(magnitude ** 2, axis=0)
            quiet_count = max(1, int(0.10 * len(frame_energies)))
            quiet_indices = np.argsort(frame_energies)[:quiet_count]
            noise_profile = np.mean(magnitude[:, quiet_indices], axis=1, keepdims=True)

            subtracted = magnitude - (alpha * noise_profile)
            gain_mask = subtracted / (magnitude + 1e-10)
            gain_mask = np.maximum(gain_mask, beta)

            smoothed_mask = gaussian_filter(gain_mask, sigma=(1.0, 1.5))
            cleaned_stft = magnitude * smoothed_mask * np.exp(1j * phase)
            _, audio = signal.istft(cleaned_stft, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def enhance_vocal_clarity_engine(audio, sr, vocal_boost_pct=50):
    """
    Dedicated AI Voice & Vocal Enhancer Engine
    Boosts speech clarity (1.5kHz-4.5kHz) and normalizes loudness to -16 LUFS.
    """
    if vocal_boost_pct > 0:
        boost_gain = (vocal_boost_pct / 100.0) * 0.6
        b_bp, a_bp = signal.butter(2, [1500 / (sr / 2), 4500 / (sr / 2)], btype='bandpass')
        speech_band = signal.filtfilt(b_bp, a_bp, audio)
        audio = audio + (speech_band * boost_gain)

    # Broadcast Loudness Normalization (-16 LUFS)
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
    current_lufs = 20 * np.log10(rms)
    gain_db = -16.0 - current_lufs
    gain_linear = 10 ** (gain_db / 20.0)

    audio = audio * gain_linear
    audio = np.tanh(audio * 0.95)
    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    try:
        return render_template('index.html')
    except Exception as e:
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
        intensity = data.get('intensity')
        
        if intensity == 'light':
            noise_pct = 40.0
        elif intensity == 'heavy':
            noise_pct = 95.0
        elif intensity == 'medium':
            noise_pct = 70.0
        else:
            noise_pct = float(data.get('denoise_pct', 70))

        if not filename:
            return jsonify({'error': 'Filename missing!'}), 400

        path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(path):
            path = os.path.join(PROCESSED_FOLDER, filename)

        if not os.path.exists(path):
            return jsonify({'error': 'File not found. Please upload again.'}), 404

        audio, sr = load_audio_ffmpeg(path)
        denoised_audio = remove_background_noise_engine(audio, sr, noise_pct)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f"denoised_{timestamp}.wav"
        out_path = os.path.join(PROCESSED_FOLDER, out_name)

        sf.write(out_path, denoised_audio, sr)
        return jsonify({'success': True, 'denoised_file': out_name})

    except Exception as e:
        logger.error(f"Denoise Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/enhance-voice', methods=['POST', 'OPTIONS'])
def handle_voice_enhancement():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        data = request.json or {}
        filename = data.get('filename')
        vocal_boost_pct = float(data.get('vocal_boost_pct', 50))

        if not filename:
            return jsonify({'error': 'Filename missing!'}), 400

        path = os.path.join(PROCESSED_FOLDER, filename)
        if not os.path.exists(path):
            path = os.path.join(UPLOAD_FOLDER, filename)

        if not os.path.exists(path):
            return jsonify({'error': 'File not found. Please upload again.'}), 404

        audio, sr = load_audio_ffmpeg(path)
        enhanced_audio = enhance_vocal_clarity_engine(audio, sr, vocal_boost_pct)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f"enhanced_{timestamp}.wav"
        out_path = os.path.join(PROCESSED_FOLDER, out_name)

        sf.write(out_path, enhanced_audio, sr)
        return jsonify({'success': True, 'denoised_file': out_name})

    except Exception as e:
        logger.error(f"Voice Enhance Error: {e}")
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
    app.run(host='127.0.0.1', port=port, debug=False)