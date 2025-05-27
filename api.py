from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from faster_whisper import WhisperModel
import tempfile
import base64
import wave
import numpy as np
from scipy import signal
import os

# ---- Model Configuration ----
MODEL_SIZE = "base.en"
DEVICE = "cpu"  # "cuda" or "cpu"
COMPUTE_TYPE = "int8"  # "float16" if DEVICE == "cuda" else "int8"

# ---- Load Model Once ----
print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Whisper model loaded successfully")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants for audio processing
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

def process_audio_data(audio_data):
    """Process audio data and return numpy array"""
    try:
        # Convert to float32 and normalize
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Apply high-pass filter to reduce low-frequency noise
        nyquist = SAMPLE_RATE / 2
        cutoff = 100  # Hz
        b, a = signal.butter(4, cutoff/nyquist, btype='high')
        audio_float = signal.filtfilt(b, a, audio_float)
        
        return audio_float
    except Exception as e:
        print(f"Error processing audio data: {str(e)}")
        return None

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    """Handle file upload transcription"""
    try:
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            segments, info = model.transcribe(tmp.name, beam_size=1)
            text = "".join([seg.text for seg in segments])
            return JSONResponse({"text": text, "language": info.language})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected.")
    
    # Use NamedTemporaryFile to buffer
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        audio_received = 0
        transcribing = False
        buffer = bytearray()
        task = None

        async def run_transcribe():
            nonlocal buffer
            try:
                # Write buffer to temp file for chunk decode
                tmp.seek(0)
                tmp.write(buffer)
                tmp.flush()
                
                # Process audio data
                audio_data = np.frombuffer(buffer, dtype=np.int16)
                processed_audio = process_audio_data(audio_data)
                
                if processed_audio is not None:
                    # Write processed audio to WAV file
                    with wave.open(tmp.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(SAMPLE_RATE)
                        wav_file.writeframes((processed_audio * 32768).astype(np.int16).tobytes())
                    
                    # Use stream() for partials
                    for segment in model.stream(tmp.name, beam_size=1):
                        if segment.text.strip():
                            print(f"Transcribed: {segment.text}")
                            await ws.send_json({
                                "type": "partial",
                                "text": segment.text,
                                "words": segment.text.split()
                            })
            except Exception as e:
                print(f"Error in transcription: {str(e)}")
                await ws.send_json({"type": "error", "error": str(e)})

        try:
            while True:
                # Receive base64 encoded audio chunk
                data = await ws.receive_json()
                if 'chunk' not in data:
                    continue
                    
                # Decode base64
                try:
                    chunk_data = data['chunk']
                    if isinstance(chunk_data, str):
                        # Add padding if needed
                        padding = len(chunk_data) % 4
                        if padding:
                            chunk_data += '=' * (4 - padding)
                        audio_chunk = base64.b64decode(chunk_data)
                    else:
                        audio_chunk = chunk_data
                        
                    buffer.extend(audio_chunk)
                    audio_received += len(audio_chunk)
                    
                    # Start transcription after receiving enough audio
                    if not transcribing and audio_received > 40960:  # ~1s of audio
                        print(f"Starting transcription after receiving {audio_received} bytes")
                        transcribing = True
                        task = asyncio.create_task(run_transcribe())
                        
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    await ws.send_json({"type": "error", "error": str(e)})
                    
        except WebSocketDisconnect:
            print("[WS] Client disconnected.")
        except Exception as e:
            print(f"WebSocket error: {str(e)}")
            await ws.send_json({"type": "error", "error": str(e)})
        finally:
            if task:
                await task

        # On disconnect: send final transcript
        try:
            if buffer:
                tmp.seek(0)
                tmp.write(buffer)
                tmp.flush()
                
                # Process final audio
                audio_data = np.frombuffer(buffer, dtype=np.int16)
                processed_audio = process_audio_data(audio_data)
                
                if processed_audio is not None:
                    # Write processed audio to WAV file
                    with wave.open(tmp.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(SAMPLE_RATE)
                        wav_file.writeframes((processed_audio * 32768).astype(np.int16).tobytes())
                    
                    segments, info = model.transcribe(tmp.name, beam_size=1)
                    text = "".join([seg.text for seg in segments])
                    await ws.send_json({
                        "type": "final",
                        "text": text,
                        "language": info.language
                    })
        except Exception as e:
            print(f"Error in final transcription: {str(e)}")
            await ws.send_json({"type": "error", "error": str(e)})
        
        await ws.close()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 