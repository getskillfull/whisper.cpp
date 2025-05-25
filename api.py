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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Configure upload folder
UPLOAD_FOLDER = '/opt/whisper/samples'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for streaming
stream_buffers = {}
stream_locks = {}
CHUNK_SIZE = 1024  # Size of audio chunks in bytes
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
MIN_AUDIO_LENGTH = 0.1  # Minimum audio length in seconds (100ms)

# Initialize whisper model
try:
    logger.info("Loading Whisper model...")
    import ssl
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

def process_audio_chunk(chunk_data, session_id):
    """Process an audio chunk and return transcription"""
    try:
        logger.info("Starting to process audio chunk...")
        # Convert base64 to bytes if needed
        if isinstance(chunk_data, str):
            logger.info("Converting base64 to bytes...")
            chunk_data = base64.b64decode(chunk_data)
        
        # Ensure chunk_data length is even
        if len(chunk_data) % 2 != 0:
            chunk_data = chunk_data[:-1]
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(chunk_data, dtype=np.int16)
        
        # Ensure audio data is not empty
        if len(audio_data) == 0:
            logger.warning("Empty audio data received")
            return None
        
        # Convert to float32 and normalize
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Apply a simple noise gate
        noise_floor = 0.01
        audio_data[np.abs(audio_data) < noise_floor] = 0
        
        # Pad audio if too short
        audio_data = pad_audio_with_silence(audio_data)
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            logger.info("Creating temporary WAV file...")
            
            # Calculate sizes
            data_size = len(audio_data) * 4  # 4 bytes per sample (float32)
            file_size = data_size + 44  # 44 is the size of the WAV header
            
            # Write WAV header
            header = create_wav_header(SAMPLE_RATE)
            # Update file size in header
            header[4:8] = (file_size - 8).to_bytes(4, 'little')
            # Update data chunk size in header
            header[40:44] = data_size.to_bytes(4, 'little')
            
            # Write header and audio data
            temp_file.write(header)
            temp_file.write(audio_data.tobytes())
            temp_file.flush()
            
            logger.info(f"Running whisper on {temp_file.name}")
            # Run whisper on the chunk with specific parameters
            result = model.transcribe(
                temp_file.name,
                language="en",
                fp16=False,  # Force FP32 since we're on CPU
                temperature=0.0,  # Reduce randomness
                best_of=1,  # Reduce computation
                beam_size=1,  # Reduce computation
                condition_on_previous_text=False,  # Don't use previous context
                no_speech_threshold=0.6  # More lenient no-speech detection
            )
            
            # Clean up
            os.unlink(temp_file.name)
            
            if result and result["text"].strip():
                text = result["text"].strip()
                # Remove repeated words
                words = text.split()
                cleaned_words = []
                for i, word in enumerate(words):
                    if i == 0 or word != words[i-1]:
                        cleaned_words.append(word)
                cleaned_text = " ".join(cleaned_words)
                
                logger.info(f"Transcription result: {cleaned_text}")
                return cleaned_text
            else:
                logger.warning("Whisper returned empty result")
                return None
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
    emit('message', {'data': 'Connected to Whisper WebSocket server.'})

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    if session_id in stream_buffers:
        del stream_buffers[session_id]
    if session_id in stream_locks:
        del stream_locks[session_id]

@socketio.on('start_stream')
def handle_start_stream(data=None):
    session_id = request.sid
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
    if session_id in stream_buffers:
        emit('final_result', {'text': 'Stream ended successfully'})

if __name__ == '__main__':
    try:
        logger.info("Starting Flask application...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.error(f"Error starting Flask application: {e}")
        raise 