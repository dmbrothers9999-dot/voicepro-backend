import os
import logging
from datetime import datetime
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import ffmpeg
import psutil

# ================= LOGGING & CONFIG SETUP =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Blogger / Web Client access allow karne ke liye

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# ================= 1. AUDIO EXPORTER CLASS =================
class AudioExporter:
    """Handles audio export to multiple formats with quality controls"""
    
    def __init__(self):
        self.supported_formats = {
            'mp3': {'codec': 'libmp3lame', 'ext': '.mp3'},
            'wav': {'codec': 'pcm_s16le', 'ext': '.wav'},
            'flac': {'codec': 'flac', 'ext': '.flac'},
            'aac': {'codec': 'aac', 'ext': '.m4a'},
            'ogg': {'codec': 'libvorbis', 'ext': '.ogg'}
        }
        
        self.bitrates = {'128': '128k', '192': '192k', '256': '256k', '320': '320k'}
        self.sample_rates = [8000, 16000, 22050, 44100, 48000, 96000]

    def export(self, input_file, output_format='mp3', bitrate='192', sample_rate=None):
        logger.info(f"📤 Exporting: format={output_format}, bitrate={bitrate}, sr={sample_rate}")
        
        output_format = output_format.lower()
        if output_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {output_format}")

        # Load audio safely
        audio, sr = librosa.load(input_file, sr=None, mono=True)
        
        # Resample if target sample rate provided
        if sample_rate and int(sample_rate) in self.sample_rates:
            target_sr = int(sample_rate)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

        # Output filename generation
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = self.supported_formats[output_format]['ext']
        output_name = f"export_{timestamp}{ext}"
        output_path = os.path.join(PROCESSED_FOLDER, output_name)

        if output_format in ['wav', 'flac']:
            # Lossless export via SoundFile
            sf.write(output_path, audio, sr)
        else:
            # Lossy export via FFmpeg
            temp_wav = os.path.join(UPLOAD_FOLDER, f"temp_{timestamp}.wav")
            sf.write(temp_wav, audio, sr)
            
            bitrate_val = self.bitrates.get(str(bitrate), '192k')
            codec = self.supported_formats[output_format]['codec']
            
            stream = ffmpeg.input(temp_wav)
            stream = ffmpeg.output(
                stream, 
                output_path, 
                acodec=codec, 
                audio_bitrate=bitrate_val, 
                ar=sr
            )
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
            
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        return output_name


# ================= 2. BACKGROUND NOISE REDUCER CLASS =================
class BackgroundNoiseReducer:
    """Advanced noise reduction algorithms"""
    
    def __init__(self):
        self.default_sr = 22050  # Memory efficient sample rate for cloud server

    def reduce_spectral_gating(self, audio, threshold_ratio=1.8):
        """Spectral Gating noise reduction"""
        D = librosa.stft(audio)
        magnitude, phase = np.abs(D), np.angle(D)
        noise_floor = np.percentile(magnitude, 15)
        
        mask = magnitude > (noise_floor * threshold_ratio)
        cleaned_mag = magnitude * mask
        return librosa.istft(cleaned_mag * np.exp(1j * phase))

    def reduce_spectral_subtraction(self, audio, noise_factor=1.5):
        """Spectral Subtraction method"""
        D = librosa.stft(audio)
        magnitude, phase = np.abs(D), np.angle(D)
        
        # Estimate noise spectrum from first 0.5s
        noise_frames = int(0.5 * self.default_sr / 512)
        noise_profile = np.mean(magnitude[:, :max(1, noise_frames)], axis=1, keepdims=True)
        
        subtracted = magnitude - (noise_factor * noise_profile)
        subtracted = np.maximum(subtracted, 0.01 * magnitude)
        return librosa.istft(subtracted * np.exp(1j * phase))

    def reduce_statistical(self, audio, reduction_strength=0.5):
        """Statistical RMS noise gate"""
        frame_len, hop_len = 2048, 512
        rms = np.array([
            np.sqrt(np.mean(audio[i:i + frame_len] ** 2))
            for i in range(0, len(audio) - frame_len, hop_len)
        ])
        
        if len(rms) == 0:
            return audio

        noise_floor = np.percentile(rms, 10)
        gate = np.ones_like(audio)
        
        for i, r in enumerate(rms):
            start = i * hop_len
            end = min(start + frame_len, len(audio))
            if r < noise_floor:
                gate[start:end] = reduction_strength
                
        return audio * gate

    def process_noise_reduction(self, input_file, method='combined', intensity='medium'):
        logger.info(f"🔇 Processing noise: method={method}, intensity={intensity}")
        
        # Load audio (mono=True and lower sample rate saves RAM)
        audio, sr = librosa.load(input_file, sr=self.default_sr, mono=True)
        
        if method == 'spectral_gating':
            thresh = 1.2 if intensity == 'light' else (1.8 if intensity == 'medium' else 2.5)
            cleaned = self.reduce_spectral_gating(audio, thresh)
        elif method == 'spectral_subtraction':
            factor = 1.0 if intensity == 'light' else (1.5 if intensity == 'medium' else 2.0)
            cleaned = self.reduce_spectral_subtraction(audio, factor)
        elif method == 'statistical':
            strength = 0.7 if intensity == 'light' else (0.5 if intensity == 'medium' else 0.2)
            cleaned = self.reduce_statistical(audio, strength)
        else:
            # Combined Method (Best Quality)
            step1 = self.reduce_spectral_subtraction(audio, noise_factor=1.2)
            thresh = 1.2 if intensity == 'light' else (1.8 if intensity == 'medium' else 2.5)
            cleaned = self.reduce_spectral_gating(step1, thresh)

        # Protect against float NaN/Inf errors & Clipping
        cleaned = np.nan_to_num(cleaned)
        cleaned = np.clip(cleaned, -1.0, 1.0)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"denoised_{method}_{timestamp}.wav"
        output_path = os.path.join(PROCESSED_FOLDER, output_name)
        
        sf.write(output_path, cleaned, sr)
        return output_name


# Initialize objects
exporter = AudioExporter()
reducer = BackgroundNoiseReducer()


# ================= 3. API ENDPOINTS =================

@app.route('/')
def home():
    """Server status check"""
    return jsonify({
        'status': 'VoicePro Server Online',
        'time': datetime.now().isoformat(),
        'developer': 'Daini Magician',
        'version': 'v0.1'
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload audio file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Clean filename spaces
        clean_filename = file.filename.replace(" ", "_")
        filepath = os.path.join(UPLOAD_FOLDER, clean_filename)
        file.save(filepath)
        
        return jsonify({
            'success': True, 
            'filename': clean_filename,
            'size_bytes': os.path.getsize(filepath)
        })
    except Exception as e:
        logger.error(f"Upload Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reduce-noise', methods=['POST'])
def handle_noise_reduction():
    """Apply background noise reduction"""
    try:
        data = request.json or {}
        filename = data.get('filename')
        method = data.get('method', 'combined')
        intensity = data.get('intensity', 'medium')

        if not filename:
            return jsonify({'error': 'Filename is required'}), 400

        path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(path):
            path = os.path.join(PROCESSED_FOLDER, filename)

        if not os.path.exists(path):
            return jsonify({'error': f'File {filename} not found on server'}), 404

        output_file = reducer.process_noise_reduction(path, method, intensity)
        return jsonify({'success': True, 'denoised_file': output_file})

    except Exception as e:
        logger.error(f"Noise Reduction Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-audio', methods=['POST'])
def handle_export():
    """Export processed audio to target format & bitrate"""
    try:
        data = request.json or {}
        filename = data.get('filename')
        fmt = data.get('format', 'mp3')
        bitrate = data.get('bitrate', '192')
        sample_rate = data.get('sample_rate', None)

        if not filename:
            return jsonify({'error': 'Filename is required'}), 400

        path = os.path.join(PROCESSED_FOLDER, filename)
        if not os.path.exists(path):
            path = os.path.join(UPLOAD_FOLDER, filename)

        if not os.path.exists(path):
            return jsonify({'error': f'File {filename} not found'}), 404

        output_file = exporter.export(path, fmt, bitrate, sample_rate)
        return jsonify({'success': True, 'output_file': output_file})

    except Exception as e:
        logger.error(f"Export Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Serve file for download with attachment headers"""
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

@app.route('/api/status', methods=['GET'])
def system_status():
    """Server system resource metrics"""
    mem = psutil.virtual_memory()
    return jsonify({
        'status': 'Online',
        'cpu_usage_percent': psutil.cpu_percent(),
        'ram_usage_percent': mem.percent,
        'ram_available_mb': round(mem.available / (1024 * 1024), 2)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)