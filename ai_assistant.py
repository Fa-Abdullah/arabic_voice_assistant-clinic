'''
+-------------------+        +-----------------------+        +------------------+        +------------------------+
|   Step 1: Install |        |  Step 2: Real-Time    |        |  Step 3: Pass    |        |  Step 4: Live Audio    |
|   Python Libraries|        |  Transcription with   |        |  Real-Time       |        |  Stream from ElevenLabs|
+-------------------+        |       AssemblyAI      |        |  Transcript to   |        |                        |
|                   |        +-----------------------+        |      OpenAI      |        +------------------------+
| - assemblyai      |                    |                    +------------------+                    |
| - openai          |                    |                             |                              |
| - elevenlabs      |                    v                             v                              v
| - mpv             |        +-----------------------+        +------------------+        +------------------------+
| - portaudio       |        |                       |        |                  |        |                        |
+-------------------+        |  AssemblyAI performs  |-------->  OpenAI generates|-------->  ElevenLabs streams   |
                             |  real-time speech-to- |        |  response based  |        |  response as live      |
                             |  text transcription   |        |  on transcription|        |  audio to the user     |
                             |                       |        |                  |        |                        |
                             +-----------------------+        +------------------+        +------------------------+

###### Step 1: Install Python libraries ######

brew install portaudio
pip install "assemblyai[extras]"
pip install elevenlabs==0.3.0b0
brew install mpv
pip install --upgrade openai
'''

from vosk import Model, KaldiRecognizer
import sounddevice as sd
import json
from openai import OpenAI
from gtts import gTTS
import os


class AI_Assistant:
    def __init__(self):
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
    )

        self.transcriber = None

        # Prompt
        self.full_transcript = [
            {"role":"system", "content":"You are a receptionist at a dental clinic. Be resourceful and efficient."},
        ]

###### Step 2: Real-Time Transcription with AssemblyAI ######
    model = Model("vosk-model-ar-mgb2-0.4")  # مثلا "vosk-model-small-ar-0.4"
    rec = KaldiRecognizer(model, 16000)
 
    def start_transcription(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(status, end="\r")
            if rec.AcceptWaveform(indata):
                result = json.loads(rec.Result())
                text = result.get("text", "")
            if text:
                self.generate_ai_response(text)

            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
                print("Listening... Press Ctrl+C to stop")
            while True:
                sd.sleep(1000)

    
    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None


    def on_close(self):
        #print("Closing Session")
        return

###### Step 3: Pass real-time transcript to OpenAI ######
    
    def generate_ai_response(self, text):
        print(f"\nPatient: {text}", end="\r\n")

        self.full_transcript.append({"role":"user", "content": text})

        response = self.openai_client.chat.completions.create(
            model="google/gemma-2-9b-it",
            messages=self.full_transcript
        )

        ai_response = response.choices[0].message.content
        self.generate_audio(ai_response)


###### Step 4: Generate audio with ElevenLabs ######

    def generate_audio(self, text):
        self.full_transcript.append({"role":"assistant", "content": text})
        print(f"\nAI Receptionist: {text}")

        tts = gTTS(text=text, lang="ar")
        filename = "ai_reply.mp3"
        tts.save(filename)
        os.system(f"start {filename}")
        # لو Mac: os.system(f"afplay {filename}")
        # لو Linux: os.system(f"mpg123 {filename}")
    
# بدء المحادثة
greeting = "Thank you for calling Vancouver dental clinic. My name is Sandy, how may I assist you?"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()

        
