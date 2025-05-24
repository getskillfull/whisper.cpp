from flask import Flask, request, jsonify, Response, stream_with_context
import os
import subprocess
import tempfile
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError
from flask_socketio import SocketIO, emit

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = '/opt/whisper/samples'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

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
    emit('message', {'data': 'Connected to Whisper WebSocket server.'})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    # data['chunk'] should be the audio bytes (base64-encoded or raw)
    # For demonstration, we'll just echo the chunk size
    # In production, you would buffer and process with a streaming model
    emit('partial_result', {'text': f"Received chunk of size {len(data['chunk'])}"})

@socketio.on('end_stream')
def handle_end_stream(data):
    # Here you would finalize transcription and send the final result
    emit('final_result', {'text': 'Transcription complete.'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000) 