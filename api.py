import asyncio
import tempfile
import base64
import wave
import numpy as np
from scipy import signal
import os
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# —— Configuration —— 
MODEL_SIZE   = "base"       # e.g. "tiny", "base", "small", "medium", "large"
DEVICE       = "cpu"        # Using CPU since Docker is configured for CPU
COMPUTE_TYPE = "int8"       # Using int8 for CPU optimization

# —— Initialize model & app —— 
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
app   = FastAPI()

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
        logger.info(f"Processing audio data of shape: {audio_data.shape}, dtype: {audio_data.dtype}")
        logger.info(f"Audio range: [{np.min(audio_data)}, {np.max(audio_data)}]")
        
        # Convert to float32 and normalize
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Apply high-pass filter to reduce low-frequency noise
        nyquist = SAMPLE_RATE / 2
        cutoff = 100  # Hz
        b, a = signal.butter(4, cutoff/nyquist, btype='high')
        audio_float = signal.filtfilt(b, a, audio_float)
        
        logger.info(f"Processed audio range: [{np.min(audio_float)}, {np.max(audio_float)}]")
        return audio_float
    except Exception as e:
        logger.error(f"Error processing audio data: {str(e)}")
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
    logger.info("[WS] Client connected.")
    
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
                logger.info(f"Audio buffer size: {len(buffer)} bytes")
                logger.info(f"Audio data shape: {audio_data.shape}")
                logger.info(f"Audio data range: [{np.min(audio_data)}, {np.max(audio_data)}]")
                
                processed_audio = process_audio_data(audio_data)
                
                if processed_audio is not None:
                    # Write processed audio to WAV file
                    with wave.open(tmp.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(SAMPLE_RATE)
                        wav_file.writeframes((processed_audio * 32768).astype(np.int16).tobytes())
                    
                    logger.info("Starting transcription...")
                    segments, info = model.transcribe(tmp.name, beam_size=1)
                    for segment in segments:
                        if segment.text.strip():
                            logger.info(f"Transcribed: {segment.text}")
                            try:
                                await ws.send_json({
                                    "type": "partial",
                                    "text": segment.text,
                                    "words": segment.text.split()
                                })
                            except Exception as e:
                                logger.error(f"Error sending transcription: {str(e)}")
                                return
                else:
                    logger.warning("No processed audio data available for transcription")
            except Exception as e:
                logger.error(f"Error in transcription: {str(e)}")
                try:
                    await ws.send_json({"type": "error", "error": str(e)})
                except Exception:
                    pass

        try:
            while True:
                # Receive raw PCM data
                msg = await ws.receive()
                
                if "bytes" in msg:
                    # Received a chunk of raw PCM audio
                    chunk = msg["bytes"]
                    logger.info(f"Received PCM chunk: {len(chunk)} bytes")
                    buffer.extend(chunk)
                    audio_received += len(chunk)
                    
                    # Start transcription after receiving enough audio
                    if not transcribing and audio_received > 8192:  # ~0.25s of audio
                        logger.info(f"Starting transcription after receiving {audio_received} bytes")
                        transcribing = True
                        task = asyncio.create_task(run_transcribe())
                elif "text" in msg:
                    # Handle text messages (e.g., stop command)
                    text = msg["text"]
                    logger.info(f"Received text message: {text}")
                    if text == "stop":
                        break
                    
        except WebSocketDisconnect:
            logger.info("[WS] Client disconnected.")
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            try:
                await ws.send_json({"type": "error", "error": str(e)})
            except Exception:
                pass
        finally:
            if task:
                try:
                    await task
                except Exception as e:
                    logger.error(f"Error in task: {str(e)}")

        # On disconnect: send final transcript
        try:
            if buffer:
                logger.info("Processing final transcription...")
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
                    logger.info(f"Final transcription: {text}")
                    try:
                        await ws.send_json({
                            "type": "final",
                            "text": text,
                            "language": info.language
                        })
                    except Exception:
                        pass
                else:
                    logger.warning("No processed audio data available for final transcription")
        except Exception as e:
            logger.error(f"Error in final transcription: {str(e)}")
            try:
                await ws.send_json({"type": "error", "error": str(e)})
            except Exception:
                pass
        
        try:
            await ws.close()
        except Exception:
            pass

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)