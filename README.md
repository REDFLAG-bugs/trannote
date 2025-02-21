# Trannote

Trannote is a real-time transcription and speaker diarization system designed to deliver high-accuracy transcriptions with minimal latency. This project leverages OpenAI's Whisper model for transcription and AssemblyAI for speaker diarization. The current implementation processes audio on the server side, but future iterations will transition to client-side processing for improved efficiency.

## Features
- **Real-time transcription** using OpenAI's Whisper model.
- **Speaker diarization** powered by AssemblyAI.
- **WebSocket-based communication** for continuous audio streaming.
- **Live text display** in a simple, user-friendly web interface.

## Tech Stack
- Python (WebSockets, asyncio)
- Whisper (OpenAI) for transcription
- AssemblyAI for diarization
- Sounddevice for capturing audio

## Future Enhancements
- Diarization to **Pyannote-Audio** once it's stable on Hugging Face.
- Optimize **transcription latency** to support near-instant results.
- Shift **audio input from server-side to client-side** in future updates.

## How It Works
1. Start the WebSocket server (`transcription.py`) to listen for audio streams.
2. The web client (`index.html`) establishes a WebSocket connection.
3. Audio is recorded and streamed from the client to the server.
4. Whisper transcribes the audio in real-time and sends text back to the client.
5. After stopping the recording, the entire audio file is sent for diarization.
6. AssemblyAI processes the file and identifies speakers.

## Setup & Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/REDFLAG-bugs/trannote.git
   cd trannote
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up the environment variable for AssemblyAI:
   ```sh
   export ASSEMBLYAI_API_KEY=your_api_key_here
   ```
4. Run the WebSocket server:
   ```sh
   python transcription.py
   ```
5. Open `index.html` in a browser and start transcribing!

## Contributing
Feel free to contribute by reporting issues, suggesting features, or submitting pull requests. The goal is to make **Trannote** a truly real-time transcription powerhouse!

## License
This project is open-source and available under the MIT License.

---
### **Next Steps**
- Improve latency and optimize for real-time performance.
- Fine-tune Whisper models for domain-specific accuracy.
- Deploy on Hugging Face Spaces for wider accessibility.

👀 Stay tuned for updates as **Trannote** evolves into a fully optimized real-time transcription solution!

