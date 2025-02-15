import asyncio
import websockets
import json
import sounddevice as sd
import numpy as np
import whisper
import wave
import os
from queue import Queue
from tempfile import NamedTemporaryFile
import assemblyai as aai

fs = 16000  
duration = 3  
device = "cpu"
model = whisper.load_model("medium", device=device)

audio_queue = Queue()
recording = False
full_transcription = ""

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

def audio_callback(indata, frames, time, status):
    if recording:
        audio_queue.put(indata.copy())

def normalize_audio(audio_data):
    return audio_data.astype(np.float32) / np.iinfo(np.int16).max

def transcribe_audio(audio_data):
    try:
        result = model.transcribe(audio_data, fp16=False, language="en", hallucination_silence_threshold=0.1, task="transcribe")
        return result.get("text", "")
    except Exception as e:
        return f"Error: {e}"

async def handle_client(websocket):
    global recording, full_transcription
    buffer = np.empty((0, 1), dtype="int16")
    audio_frames = [] 

    try:
        while True:
            message = await websocket.recv()

            if message == "start":
                print("Recording started...")
                recording = True
                with sd.InputStream(samplerate=fs, channels=1, dtype="int16", callback=audio_callback):
                    while recording:
                        while not audio_queue.empty():
                            chunk = audio_queue.get()
                            buffer = np.append(buffer, chunk, axis=0)
                            audio_frames.append(chunk)

                        if len(buffer) >= fs * duration:
                            chunk = buffer[:fs * duration]
                            buffer = buffer[fs * duration:]
                            normalized_audio = normalize_audio(chunk.flatten())
                            transcription = transcribe_audio(normalized_audio)
                            full_transcription += transcription + " "

                            await websocket.send(json.dumps({
                                "status": "success",
                                "transcription": transcription,
                                "full_transcription": full_transcription
                            }))

                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                            if message == "stop":
                                recording = False
                                print("Recording stopped.")
                                break
                        except asyncio.TimeoutError:
                            pass

            if message == "stop":
                recording = False
                print("Processing final transcription...")

                while len(buffer) > 0:
                    chunk = buffer[:fs * duration]
                    buffer = buffer[fs * duration:]
                    normalized_audio = normalize_audio(chunk.flatten())
                    transcription = transcribe_audio(normalized_audio)
                    full_transcription += transcription + " "

                    await websocket.send(json.dumps({
                        "status": "successAfter(STOP)",
                        "transcription": transcription,
                        "full_transcription": full_transcription,
                        "final_chunk": True
                    }))

                buffer = np.empty((0, 1), dtype="int16")

                # Save recorded audio to temp file
                with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    with wave.open(temp_audio.name, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(fs)
                        wf.writeframes(np.concatenate(audio_frames).tobytes())

                    print(f"Saved audio for diarization: {temp_audio.name}")

                await websocket.send(json.dumps({
                    "status": "diarization_started",
                    "message": "Identifying speakers..."
                }))

                diarization_text = await get_diarization(temp_audio.name)

                await websocket.send(json.dumps({
                    "status": "diarization_complete",
                    "diarization": diarization_text
                }))

                os.remove(temp_audio.name)
                break

    except websockets.ConnectionClosed:
        print("Client disconnected.")

    except Exception as e:
        print(f"Error: {e}")

async def get_diarization(audio_path):
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=config)

    diarized_text = "\n".join([f"Speaker {utt.speaker}: {utt.text}" for utt in transcript.utterances])
    return diarized_text

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        print("Server running on ws://<IP>:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())