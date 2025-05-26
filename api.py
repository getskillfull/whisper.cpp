from flask import Flask, request, jsonify
import os
import subprocess
import tempfile
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError
from flask_socketio import SocketIO, emit
import base64
import wave
import io
import numpy as np
from threading import Lock
import queue
import time
import logging
import whisper
import ssl
from scipy import signal
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=60, ping_interval=25)

# Configure upload folder
UPLOAD_FOLDER = '/opt/whisper/samples'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for streaming
stream_buffers = {}
stream_locks = {}
audio_buffers = {}

# Constants
CHUNK_SIZE = 1024
SAMPLE_RATE = 16000
MIN_AUDIO_LENGTH = 0.1  # Minimum audio length to process (seconds)
NOISE_FLOOR = 0.005  # Lower noise floor threshold
MIN_SPEECH_DURATION = 0.1  # Minimum speech duration (seconds)
MAX_BUFFER_DURATION = 0.3  # Maximum buffer duration (seconds)
MIN_BUFFER_DURATION = 0.15  # Minimum buffer duration (seconds)
BUFFER_TIMEOUT = 0.2  # Buffer timeout (seconds)

# Buffer for each session
last_buffer_time = {}

# Initialize whisper model
try:
    logger.info("Loading Whisper model...")
    ssl._create_default_https_context = ssl._create_unverified_context
    model = whisper.load_model("base.en")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

def create_wav_header(sample_rate, channels=1, sample_width=4):
    """Create a WAV header for the given parameters"""
    header = bytearray()
    # RIFF header
    header.extend(b'RIFF')
    header.extend((0).to_bytes(4, 'little'))  # File size - 8 (to be filled later)
    header.extend(b'WAVE')
    # fmt chunk
    header.extend(b'fmt ')
    header.extend((16).to_bytes(4, 'little'))  # fmt chunk size
    header.extend((3).to_bytes(2, 'little'))   # Audio format (3 for float32)
    header.extend((channels).to_bytes(2, 'little'))  # Number of channels
    header.extend((sample_rate).to_bytes(4, 'little'))  # Sample rate
    header.extend((sample_rate * channels * sample_width).to_bytes(4, 'little'))  # Byte rate
    header.extend((channels * sample_width).to_bytes(2, 'little'))  # Block align
    header.extend((sample_width * 8).to_bytes(2, 'little'))  # Bits per sample
    # data chunk
    header.extend(b'data')
    header.extend((0).to_bytes(4, 'little'))  # Data chunk size (to be filled later)
    return header

def pad_audio_with_silence(audio_data, target_length_ms=1000):
    """Pad audio data with silence to reach target length"""
    current_length_ms = len(audio_data) / SAMPLE_RATE * 1000
    if current_length_ms >= target_length_ms:
        return audio_data
        
    # Calculate number of silence samples needed
    silence_samples = int((target_length_ms - current_length_ms) * SAMPLE_RATE / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)
    
    # Add silence to both ends for better context
    half_silence = silence_samples // 2
    return np.concatenate([
        np.zeros(half_silence, dtype=np.float32),
        audio_data,
        np.zeros(silence_samples - half_silence, dtype=np.float32)
    ])

def buffer_audio_chunk(chunk_data, session_id):
    """Buffer audio chunk and process when enough data is available."""
    try:
        if session_id not in audio_buffers:
            audio_buffers[session_id] = []
            stream_buffers[session_id] = []
            stream_locks[session_id] = threading.Lock()
        
        with stream_locks[session_id]:
            # Convert base64 to bytes
            try:
                # Add padding if needed
                padding = 4 - (len(chunk_data) % 4)
                if padding != 4:
                    chunk_data = chunk_data + ('=' * padding)
                
                audio_bytes = base64.b64decode(chunk_data)
                
                # Ensure the audio data length is even
                if len(audio_bytes) % 2 != 0:
                    audio_bytes = audio_bytes[:-1]
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Log audio data details
                logger.info(f"Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
                logger.info(f"Audio data range: [{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
                
                # Convert to float32 and normalize
                audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Apply noise gate
                audio_data[np.abs(audio_data) < NOISE_FLOOR] = 0
                logger.info(f"Non-zero samples after noise gate: {np.count_nonzero(audio_data)}")
                
                # Add to buffer
                audio_buffers[session_id].append(audio_data)
                
                # Calculate total duration
                total_duration = sum(len(chunk) for chunk in audio_buffers[session_id]) / SAMPLE_RATE
                logger.info(f"Total buffered duration: {total_duration:.3f}s")
                
                # Process if we have enough data
                if total_duration >= MIN_BUFFER_DURATION:
                    # Combine all chunks
                    combined_audio = np.concatenate(audio_buffers[session_id])
                    
                    # Process the audio
                    result = process_audio_data(combined_audio, session_id)
                    
                    # Keep the last chunk if it's recent
                    if len(audio_buffers[session_id]) > 1:
                        audio_buffers[session_id] = [audio_buffers[session_id][-1]]
                    else:
                        audio_buffers[session_id] = []
                    
                    return result
                else:
                    logger.info(f"Buffering audio chunk, current duration: {total_duration:.3f}s")
                    return None
                    
            except Exception as e:
                logger.error(f"Error processing audio data: {str(e)}")
                return None
                
    except Exception as e:
        logger.error(f"Error buffering audio chunk: {str(e)}")
        return None

def is_silence(audio_data, sample_rate):
    """Check if audio segment is silence."""
    # Calculate RMS energy
    rms = np.sqrt(np.mean(np.square(audio_data)))
    logger.info(f"Audio RMS energy: {rms:.6f}")
    
    # Count non-zero samples
    non_zero = np.count_nonzero(np.abs(audio_data) > NOISE_FLOOR)
    speech_duration = non_zero / sample_rate
    logger.info(f"Speech duration: {speech_duration:.3f}s (non-zero samples: {non_zero})")
    
    # Check if duration is too short
    if speech_duration < MIN_SPEECH_DURATION:
        logger.info(f"Speech duration {speech_duration:.3f}s below minimum {MIN_SPEECH_DURATION}s")
        return True
    
    # Check if energy is too low
    if rms < NOISE_FLOOR:
        logger.info(f"RMS energy {rms:.6f} below noise floor {NOISE_FLOOR}")
        return True
    
    return False

def process_buffered_audio(session_id):
    """Process any remaining audio in the buffer"""
    if session_id not in audio_buffers or not audio_buffers[session_id]:
        return None
    
    # Calculate total duration of buffered audio
    total_samples = sum(len(chunk) for chunk in audio_buffers[session_id])
    total_duration = total_samples / SAMPLE_RATE
    
    # Only process if we have minimum duration
    if total_duration >= MIN_BUFFER_DURATION:
        combined_audio = np.concatenate(audio_buffers[session_id])
        audio_buffers[session_id] = []  # Clear buffer
        return process_audio_data(combined_audio, session_id)
    
    return None

def emit_word(session_id, word):
    """Emit a single word to the client"""
    try:
        socketio.emit('word', {'text': word}, room=session_id)
    except Exception as e:
        logger.error(f"Error emitting word: {e}")

def process_audio_data(audio_data, session_id):
    """Process audio data and return transcription."""
    try:
        # Apply high-pass filter to reduce low-frequency noise
        nyquist = SAMPLE_RATE / 2
        cutoff = 100  # Hz
        b, a = signal.butter(4, cutoff/nyquist, btype='high')
        audio_data = signal.filtfilt(b, a, audio_data)
        logger.info("Applied high-pass filter")
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes((audio_data * 32768.0).astype(np.int16).tobytes())
        
        logger.info(f"Running whisper on {temp_filename}")
        
        # Run whisper with more lenient parameters
        result = model.transcribe(
            temp_filename,
            language="en",
            task="transcribe",
            fp16=False,
            no_speech_threshold=0.3,  # Even more lenient no-speech detection
            logprob_threshold=-1.0,   # More lenient log probability threshold
            compression_ratio_threshold=2.4,
            condition_on_previous_text=True,
            initial_prompt="Transcribe the following audio:",
            temperature=0.0,  # Reduce randomness
            best_of=1,  # Reduce computation
            beam_size=1  # Reduce computation
        )
        
        # Clean up temporary file
        os.unlink(temp_filename)
        
        if not result["text"].strip():
            logger.warning("Whisper returned empty result")
            return None
            
        # Split into words and emit each word
        words = result["text"].strip().split()
        for word in words:
            emit_word(session_id, word)
            # Add a small delay between word emissions
            time.sleep(0.05)
            
        return result["text"].strip()
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None

def process_audio_chunk(chunk_data, session_id):
    """Process an audio chunk and return transcription"""
    try:
        logger.info("Starting to process audio chunk...")
        
        # Ensure chunk_data is a string
        if not isinstance(chunk_data, str):
            chunk_data = chunk_data.decode('utf-8')
        
        logger.info("Converting base64 to bytes...")
        
        # Process the audio chunk
        result = buffer_audio_chunk(chunk_data, session_id)
        if result:
            logger.info(f"Transcription result: {result}")
            emit('partial_result', {'text': result})
        
        return result
        
    except Exception as e:
        logger.error(f"Error in process_audio_chunk: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_from_s3(bucket_name, object_key, local_path):
    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, object_key, local_path)
        return True
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        return False

def generate_presigned_url(bucket_name, object_key, expiration=3600):
    try:
        s3_client = boto3.client('s3')
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_key
            },
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")
        return None

@app.route('/get-s3-url', methods=['POST'])
def get_s3_url():
    if 's3_path' not in request.form:
        return jsonify({'error': 'No S3 path provided'}), 400
    
    try:
        s3_path = request.form['s3_path']
        bucket_name = s3_path.split('/')[0]
        object_key = '/'.join(s3_path.split('/')[1:])
        
        url = generate_presigned_url(bucket_name, object_key)
        if url:
            return jsonify({
                'url': url,
                'expires_in': 3600
            })
        else:
            return jsonify({'error': 'Failed to generate URL'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files and 's3_path' not in request.form:
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        if 'file' in request.files:
            # Handle direct file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
        else:
            # Handle S3 file
            s3_path = request.form['s3_path']
            bucket_name = s3_path.split('/')[0]
            object_key = '/'.join(s3_path.split('/')[1:])
            filename = os.path.basename(object_key)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if not download_from_s3(bucket_name, object_key, filepath):
                return jsonify({'error': 'Failed to download file from S3'}), 500
        
        # Run whisper-cli
        result = subprocess.run([
            '/app/build/bin/whisper-cli',
            '-m', '/app/models/ggml-base.en.bin',
            '-f', filepath
        ], capture_output=True, text=True)
        
        # Clean up the file
        os.remove(filepath)
        
        if result.returncode == 0:
            return jsonify({
                'transcription': result.stdout,
                'status': 'success'
            })
        else:
            return jsonify({
                'error': 'Transcription failed',
                'details': result.stderr
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    stream_buffers[session_id] = queue.Queue()
    stream_locks[session_id] = Lock()
    logger.info(f"Client connected: {session_id}")
    emit('message', {'data': 'Connected to Whisper WebSocket server.'})

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    logger.info(f"Client disconnected: {session_id}")
    if session_id in stream_buffers:
        del stream_buffers[session_id]
    if session_id in stream_locks:
        del stream_locks[session_id]
    if session_id in audio_buffers:
        del audio_buffers[session_id]
    if session_id in last_buffer_time:
        del last_buffer_time[session_id]

@socketio.on('start_stream')
def handle_start_stream(data=None):
    session_id = request.sid
    logger.info(f"Stream started for session: {session_id}")
    emit('stream_started', {'message': 'Stream started successfully'})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    session_id = request.sid
    logger.info(f"Received audio chunk from session {session_id}")
    
    if session_id not in stream_buffers:
        logger.error(f"Stream not initialized for session {session_id}")
        emit('error', {'message': 'Stream not initialized'})
        return
    
    try:
        # Process the chunk and get transcription
        logger.info("Processing audio chunk...")
        transcription = process_audio_chunk(data['chunk'], session_id)
        if transcription:
            logger.info(f"Transcription result: {transcription}")
            emit('partial_result', {'text': transcription})
        else:
            logger.warning("No transcription result received")
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        emit('error', {'message': f'Error processing chunk: {str(e)}'})

@socketio.on('end_stream')
def handle_end_stream(data=None):
    session_id = request.sid
    logger.info(f"Stream ended for session: {session_id}")
    
    # Process any remaining audio in the buffer
    transcription = process_buffered_audio(session_id)
    if transcription:
        emit('partial_result', {'text': transcription})
    
    emit('final_result', {'text': 'Stream ended successfully'})

if __name__ == '__main__':
    try:
        logger.info("Starting Flask application...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting Flask application: {e}")
        raise 