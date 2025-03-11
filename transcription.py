import asyncio
import websockets
import json
import whisper
import os
import tempfile
from io import BytesIO
from dataclasses import dataclass
import logging
import concurrent.futures
from collections import deque
import time
import subprocess
import wave
import struct

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    translated_text: str = ""

def convert_webm_to_wav(input_file, output_file, logger):
    try:
        with open(input_file, 'rb') as f:
            header = f.read(12)
            if header.startswith(b'RIFF') and b'WAVE' in header:
                logger.info(f"File is already in WAV format, copying: {input_file} -> {output_file}")
                with open(output_file, 'wb') as out_f:
                    f.seek(0)
                    out_f.write(f.read())
                return True, ""
        cmd = [
            "ffmpeg",
            "-v", "warning",
            "-f", "webm",
            "-i", input_file,
            "-map", "0:a:0",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-f", "wav",
            "-y",
            output_file
        ]

        logger.info(f"Converting audio with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            return False, result.stderr

        logger.info(f"Audio conversion successful: {input_file} -> {output_file}")
        return True, ""

    except Exception as e:
        error_msg = f"Exception during conversion: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

class AudioProcessor:
    def __init__(self):
        logger.info("Loading Whisper model...")
        self.model = whisper.load_model("medium")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        logger.info("Whisper model loaded successfully")

    async def transcribe_audio(self, audio_data: BytesIO, is_final: bool = False, prompt: str = "") -> TranscriptionResult:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                self._transcribe_sync, 
                audio_data.getvalue(), 
                is_final,
                prompt
            )
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise

    def _transcribe_sync(self, audio_bytes: bytes, is_final: bool = False, prompt: str = "") -> TranscriptionResult:
        start_time = time.time()
        
        temp_input = None
        temp_output = None
        
        try:
            is_wav = audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:12]
            
            if is_wav:
                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_output = temp_input 
            else:
                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_output.close()
            
            input_path = temp_input.name
            output_path = temp_output.name if temp_output != temp_input else input_path
            
            temp_input.close()
            
            with open(input_path, 'wb') as f:
                f.write(audio_bytes)
            
            logger.info(f"Audio chunk size: {len(audio_bytes)} bytes")
            if len(audio_bytes) > 20:
                hex_sample = audio_bytes[:20].hex()
                logger.info(f"First 20 bytes of audio chunk: {hex_sample}")
            if not is_wav:
                success, error_msg = convert_webm_to_wav(input_path, output_path, logger)
                
                if not success:
                    raise Exception(f"Audio conversion failed: {error_msg}")
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    raise Exception("Conversion produced empty output file")
            else:
                logger.info(f"Using WAV file directly without conversion: {input_path}")
            
            logger.info(f"Processing audio with Whisper...")
            
            try:
                with wave.open(output_path, 'rb') as wav_file:
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    framerate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    
                    logger.info(f"WAV file info: channels={channels}, sample_width={sample_width}, "
                                f"framerate={framerate}, frames={n_frames}")
                    
                    if n_frames == 0:
                        raise Exception("WAV file contains no audio frames")
            except Exception as wav_error:
                logger.error(f"Invalid WAV file: {str(wav_error)}")
                raise Exception(f"Invalid WAV file: {str(wav_error)}")
          
            should_detect_language = is_final or len(audio_bytes) > 10000  
            
            try:
                if should_detect_language:
                    detection_result = self.model.detect_language(output_path)[0]
                    detected_language = detection_result["language"]
                    language_prob = detection_result["probability"]
                    logger.info(f"Detected language: {detected_language} (probability: {language_prob:.2f})")
                else:
                    detected_language = "en"
                    language_prob = 0.8
                    logger.info(f"Skipping language detection for small chunk, using default: {detected_language}")
            except Exception as lang_error:
                logger.error(f"Language detection failed: {str(lang_error)}")
                detected_language = "en"
                language_prob = 0.5
                logger.info(f"Defaulting to language: {detected_language}")
            if is_final:
                beam_size = 5
                best_of = 5
                temperature = 0.0
            else:
                beam_size = 1
                best_of = 1
                temperature = 0.2
            
            try:
                result = self.model.transcribe(
                    output_path,
                    task="translate" if detected_language != "en" else "transcribe",
                    language=detected_language, 
                    beam_size=beam_size,
                    best_of=best_of,
                    temperature=temperature,
                    hallucination_silence_threshold=0.1,
                    prompt=prompt if prompt else None
                )
            except Exception as transcribe_error:
                logger.error(f"Transcription failed: {str(transcribe_error)}")
                return TranscriptionResult(
                    text="[Transcription failed]",
                    language=detected_language,
                    confidence=language_prob
                )

            elapsed = time.time() - start_time
            logger.info(f"Transcription took {elapsed:.2f} seconds" + (" (final)" if is_final else ""))
            original_text = ""
            if detected_language != "en" and (is_final or len(audio_bytes) > 10000):
                try:
                    original_result = self.model.transcribe(
                        output_path,
                        task="transcribe", 
                        language=detected_language,
                        beam_size=3 if is_final else 1,
                        best_of=3 if is_final else 1,
                        temperature=0.0
                    )
                    original_text = original_result.get("text", "").strip()
                    logger.info(f"Original transcription ({detected_language}): '{original_text}'")
                except Exception as orig_error:
                    logger.error(f"Original language transcription failed: {str(orig_error)}")

            translated_text = result.get("text", "").strip() or "[No speech detected]"
            return TranscriptionResult(
                text=translated_text, 
                language=detected_language,
                confidence=language_prob,
                translated_text=original_text if detected_language != "en" else "" 
            )
            
        except Exception as e:
            logger.error(f"Transcription processing error: {str(e)}")
            raise
        finally:
            for temp_file in [temp_input, temp_output]:
                if temp_file is not None:
                    try:
                        path = temp_file.name
                        if os.path.exists(path):
                            os.unlink(path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file: {str(e)}")

class TranscriptionServer:
    def __init__(self, max_history_chunks=50):
        self.audio_processor = AudioProcessor()
        self.max_history_chunks = max_history_chunks
        
    async def handle_client(self, websocket):
        audio_chunks = deque(maxlen=self.max_history_chunks)
        running_transcription = ""
        original_transcription = ""
        detected_language = "en"
        chunk_count = 0
        previous_chunk_timestamps = {}
        
        try:
            logger.info("New client connected")
            await websocket.send(json.dumps({
                "status": "connection_ready",
                "message": "Server ready for audio data"
            }))
            
            async for message in websocket:
                if isinstance(message, bytes):
                    chunk_count += 1
                    if len(message) < 100:
                        logger.warning(f"Received empty or invalid audio chunk #{chunk_count}")
                        continue

                    logger.info(f"Processing audio chunk #{chunk_count} ({len(message)} bytes)")
                    audio_chunks.append(message)
                    try:
                        await websocket.send(json.dumps({
                            "status": "processing",
                            "message": f"Processing chunk #{chunk_count}..."
                        }))
                        chunk_audio = BytesIO(message)
                        prompt = self._get_prompt_from_transcription(running_transcription)
                        chunk_result = await self.audio_processor.transcribe_audio(
                            chunk_audio, 
                            prompt=prompt
                        )
                        if chunk_result.confidence > 0.6: 
                            detected_language = chunk_result.language
                        if chunk_result.text and chunk_result.text != "[No speech detected]":

                            if not self._is_duplicate(running_transcription, chunk_result.text):
                                if running_transcription:
                                    running_transcription += " "
                                    
                                running_transcription += chunk_result.text
                                logger.info(f"Chunk #{chunk_count} transcription: '{chunk_result.text}'")
                                logger.info(f"Updated running transcription: '{running_transcription}'")
                                if chunk_result.translated_text:
                                    if original_transcription:
                                        original_transcription += " "
                                    original_transcription += chunk_result.translated_text
                                    logger.info(f"Updated original transcription: '{original_transcription}'")
                                await websocket.send(json.dumps({
                                    "status": "success",
                                    "chunk_transcription": chunk_result.text,
                                    "full_transcription": running_transcription,
                                    "original_language": detected_language,
                                    "original_transcription": original_transcription if detected_language != "en" else "",
                                    "chunk_count": chunk_count
                                }))
                            else:
                                logger.info(f"Skipping duplicate chunk: '{chunk_result.text}'")
                                await websocket.send(json.dumps({
                                    "status": "success",
                                    "chunk_transcription": "[Duplicate content filtered]",
                                    "full_transcription": running_transcription,
                                    "original_language": detected_language,
                                    "original_transcription": original_transcription if detected_language != "en" else "",
                                    "chunk_count": chunk_count
                                }))
                        else:
                            logger.info(f"No speech detected in chunk #{chunk_count}")
                            await websocket.send(json.dumps({
                                "status": "success",
                                "chunk_transcription": "[No speech detected]",
                                "full_transcription": running_transcription,
                                "original_language": detected_language,
                                "original_transcription": original_transcription if detected_language != "en" else "",
                                "chunk_count": chunk_count
                            }))
                        await asyncio.sleep(0.05)
                        await websocket.send(json.dumps({
                            "status": "chunk_processed",
                            "chunk_count": chunk_count,
                            "transcription_length": len(running_transcription)
                        }))
                        
                    except Exception as chunk_error:
                        logger.error(f"Error processing chunk #{chunk_count}: {str(chunk_error)}")
                        await websocket.send(json.dumps({
                            "status": "chunk_error",
                            "message": f"Error processing chunk #{chunk_count}, continuing with next chunk",
                            "error": str(chunk_error)
                        }))
                        continue

                else:
                    try:
                        data = json.loads(message)
                        if data["action"] == "stop":
                            logger.info(f"Processing final transcription with {len(audio_chunks)} stored chunks...")
                            if audio_chunks:
                                try:
                                    await websocket.send(json.dumps({
                                        "status": "processing",
                                        "message": "Processing final high-quality transcription..."
                                    }))
                                    combined_audio = self._combine_audio_chunks(audio_chunks)
                                    
                                    if combined_audio.getvalue():
                                        logger.info(f"Combined audio size: {len(combined_audio.getvalue())} bytes")
                                        final_result = await self.audio_processor.transcribe_audio(
                                            combined_audio,
                                            is_final=True
                                        )
                                        
                                        logger.info(f"Final transcription completed: '{final_result.text}'")
                                        await websocket.send(json.dumps({
                                            "status": "final_transcription",
                                            "full_transcription": final_result.text,
                                            "original_language": final_result.language,
                                            "original_transcription": final_result.translated_text if final_result.language != "en" else ""
                                        }))
                                except Exception as final_error:
                                    logger.error(f"Error processing final transcription: {str(final_error)}")
                                    await websocket.send(json.dumps({
                                        "status": "error",
                                        "message": f"Error processing final transcription: {str(final_error)}"
                                    }))
                            await websocket.send(json.dumps({
                                "status": "processing_complete",
                                "message": "Audio processing completed"
                            }))
                            logger.info("Processing completed, closing connection")
                            break
                        elif data["action"] == "ping":
                            await websocket.send(json.dumps({
                                "status": "pong"
                            }))
                            
                    except json.JSONDecodeError:
                        logger.error("Received invalid JSON message")
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": "Invalid JSON message"
                        }))
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": f"Error processing message: {str(e)}"
                        }))
        except Exception as e:
            logger.error(f"Error in client connection: {str(e)}")
        finally:
            logger.info("Client disconnected")
            
    def _get_prompt_from_transcription(self, transcription, max_words=10):
        """Extract the last few words from the transcription to use as a prompt."""
        if not transcription:
            return ""
        
        words = transcription.split()
        if len(words) <= max_words:
            return transcription
        
        return " ".join(words[-max_words:])
    
    def _is_duplicate(self, full_text, new_chunk, similarity_threshold=0.7):
        logger.info(f"Checking if '{new_chunk}' is a duplicate in '{full_text}'")
        if not new_chunk or len(new_chunk.split()) <= 1:
            logger.info("Chunk is empty or too short, not a duplicate")
            return False
        words_to_check = 10
        full_words = full_text.split()
        
        if len(full_words) >= words_to_check:
            last_words = ' '.join(full_words[-words_to_check:])
            overlap = 0
            for word in new_chunk.split():
                if word.lower() in last_words.lower():
                    overlap += 1
            
            overlap_ratio = overlap / max(1, len(new_chunk.split()))
            logger.info(f"Overlap ratio: {overlap_ratio} (threshold: {similarity_threshold})")
            
            if overlap_ratio > similarity_threshold:
                logger.info("Detected as duplicate")
                return True
                
        logger.info("Not a duplicate")
        return False
    
    def _combine_audio_chunks(self, audio_chunks):
        # Check if the chunks are WAV files by examining the first chunk
        if not audio_chunks:
            return BytesIO()
        
        first_chunk = audio_chunks[0]
        is_wav = first_chunk.startswith(b'RIFF') and b'WAVE' in first_chunk[:12]
        
        if is_wav:
            # For WAV files, we need to extract and combine the audio data
            logger.info("Combining WAV audio chunks...")
            
            all_audio_data = []
            wav_params = None
            
            # Extract audio data from each WAV chunk
            for i, chunk in enumerate(audio_chunks):
                try:
                    wav_io = BytesIO(chunk)
                    with wave.open(wav_io, 'rb') as wav_file:
                        # Store parameters from the first valid WAV chunk
                        if wav_params is None:
                            wav_params = wav_file.getparams()
                        
                        # Read audio frames
                        audio_data = wav_file.readframes(wav_file.getnframes())
                        all_audio_data.append(audio_data)
                        
                        logger.info(f"Extracted {len(audio_data)} bytes of audio data from chunk {i+1}")
                except Exception as e:
                    logger.error(f"Error processing WAV chunk {i+1}: {str(e)}")
                    # Skip invalid chunks
                    continue
            
            # If we couldn't extract any audio data, return an empty buffer
            if not all_audio_data or wav_params is None:
                logger.warning("Failed to extract audio data from WAV chunks")
                return BytesIO()
            
            # Combine all audio data
            combined_data = b''.join(all_audio_data)
            
            # Create a new WAV file with combined data
            output = BytesIO()
            with wave.open(output, 'wb') as out_wav:
                out_wav.setparams(wav_params)
                out_wav.writeframes(combined_data)
            
            output.seek(0)
            logger.info(f"Successfully combined WAV chunks: total size {len(output.getvalue())} bytes")
            return output
        else:
            # For non-WAV files (WebM), just concatenate them
            logger.info("Combining non-WAV audio chunks...")
            combined = BytesIO()
            for chunk in audio_chunks:
                combined.write(chunk)
            combined.seek(0)
            return combined

async def main():
    server = TranscriptionServer()
    
    async with websockets.serve(
        server.handle_client,
        "0.0.0.0",
        8765,
        ping_interval=20,
        ping_timeout=60,
        max_size=10 * 1024 * 1024 
    ):
        logger.info("Server started on ws://localhost:8765")
        logger.info("Waiting for connections...")
        await asyncio.Future() 

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown initiated")