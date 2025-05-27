import asyncio
import tempfile

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# —— Configuration —— 
MODEL_SIZE   = "base"       # e.g. "tiny", "base", "small", "medium", "large"
DEVICE       = "cuda"       # or "cpu"
COMPUTE_TYPE = "float16"    # depends on your setup

# —— Initialize model & app —— 
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
app   = FastAPI()


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    """
    Synchronous file upload endpoint.
    Returns full transcription and detected language.
    """
    # Buffer the upload to disk
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()

        # Run full transcription
        segments, info = model.transcribe(tmp.name, beam_size=1)
        text = "".join(seg.text for seg in segments)

        return JSONResponse({"text": text, "language": info.language})


@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    """
    Bi-directional streaming endpoint.
    Client sends raw PCM bytes over WS.
    Server streams back partial & final JSON messages.
    """
    await ws.accept()
    print("[WS] Client connected.")

    # Temporary file to accumulate audio
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        buffer = bytearray()
        audio_received = 0
        transcribing   = False
        task           = None

        async def run_transcribe():
            """
            Background task that takes current buffer
            and streams partials back to the client.
            """
            nonlocal buffer
            tmp.seek(0)
            tmp.write(buffer)
            tmp.flush()
            for segment in model.stream(tmp.name, beam_size=1):
                await ws.send_json({"type": "partial", "text": segment.text})

        try:
            while True:
                # This handles both binary and text frames
                msg = await ws.receive()

                if "bytes" in msg:
                    # Received a chunk of raw PCM audio
                    chunk = msg["bytes"]
                    buffer.extend(chunk)
                    audio_received += len(chunk)

                    # After ~40 KB (~1s), start partial streaming
                    if not transcribing and audio_received > 40_960:
                        transcribing = True
                        task = asyncio.create_task(run_transcribe())

                elif "text" in msg:
                    # Ignore any stray text frames
                    continue

        except WebSocketDisconnect:
            print("[WS] Client disconnected.")
        except Exception as e:
            # Forward the error to the client
            await ws.send_json({"type": "error", "error": str(e)})
        finally:
            # Ensure background task completes
            if task:
                try:
                    await task
                except Exception as e:
                    await ws.send_json({"type": "error", "error": str(e)})

        # Once client closes, send the final transcript
        tmp.seek(0)
        tmp.write(buffer)
        tmp.flush()
        segments, info = model.transcribe(tmp.name, beam_size=1)
        full_text = "".join(seg.text for seg in segments)

        await ws.send_json({
            "type":     "final",
            "text":     full_text,
            "language": info.language
        })
        await ws.close()
        print("[WS] Connection fully closed.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, log_level="info")