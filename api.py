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

def create_wav_header(sample_rate, channels=1, sample_width=2):
    """Create a WAV header for the given parameters"""
    header = bytearray()
    # RIFF header
    header.extend(b'RIFF')
    header.extend((0).to_bytes(4, 'little'))  # File size - 8
    header.extend(b'WAVE')
    # fmt chunk
    header.extend(b'fmt ')
    header.extend((16).to_bytes(4, 'little'))  # fmt chunk size
    header.extend((1).to_bytes(2, 'little'))   # Audio format (1 for PCM)
    header.extend((channels).to_bytes(2, 'little'))  # Number of channels
    header.extend((sample_rate).to_bytes(4, 'little'))  # Sample rate
    header.extend((sample_rate * channels * sample_width).to_bytes(4, 'little'))  # Byte rate
    header.extend((channels * sample_width).to_bytes(2, 'little'))  # Block align
    header.extend((sample_width * 8).to_bytes(2, 'little'))  # Bits per sample
    # data chunk
    header.extend(b'data')
    header.extend((0).to_bytes(4, 'little'))  # Data chunk size
    return header

def process_audio_chunk(chunk_data, session_id):
    """Process an audio chunk and return transcription"""
    try:
        # Convert base64 to bytes if needed
        if isinstance(chunk_data, str):
            chunk_data = base64.b64decode(chunk_data)
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Write WAV header
            temp_file.write(create_wav_header(SAMPLE_RATE))
            # Write audio data
            temp_file.write(chunk_data)
            temp_file.flush()
            
            # Run whisper-cli on the chunk
            result = subprocess.run([
                '/app/build/bin/whisper-cli',
                '-m', '/app/models/ggml-base.en.bin',
                '-f', temp_file.name
            ], capture_output=True, text=True)
            
            # Clean up
            os.unlink(temp_file.name)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None
    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
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
    if session_id not in stream_buffers:
        emit('error', {'message': 'Stream not initialized'})
        return
    
    try:
        # Process the chunk and get transcription
        transcription = process_audio_chunk(data['chunk'], session_id)
        if transcription:
            emit('partial_result', {'text': transcription})
    except Exception as e:
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