"""
AI Voice Assistant - Dental Clinic Receptionist
Supports Arabic speech recognition and TTS
"""

from vosk import Model, KaldiRecognizer
import sounddevice as sd
import json
from openai import OpenAI
from gtts import gTTS
import os
import time
import numpy as np
import threading

class AI_Assistant:
    def __init__(self):
        """Initialize the AI Assistant with all necessary components"""
        print("🚀 Initializing AI Assistant...")
        
        # OpenAI client setup
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )

        # Initialize Vosk model and recognizer
        model_path = self.find_model_path()
        print(f"🔧 Loading Arabic model from {model_path}...")
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, 16000)
        print("✅ Model loaded successfully!")
        
        # Control variables
        self.is_listening = False
        self.is_processing = False
        
        # Conversation history
        self.full_transcript = [
            {"role": "system", "content": """You are Sandy, a friendly and professional receptionist at Vancouver Dental Clinic. 

Your responsibilities:
- Greet callers warmly
- Schedule appointments
- Answer questions about services
- Provide clinic information
- Handle cancellations and rescheduling
- Be helpful and efficient

Clinic Info:
- Open: Monday-Friday 8AM-6PM, Saturday 9AM-3PM
- Services: General dentistry, cleanings, fillings, crowns, root canals
- Location: Downtown Vancouver
- Phone: (604) 555-DENTAL

Keep responses short and professional. Always ask how you can help further."""},
        ]

    def find_model_path(self):
        """Find the Vosk model folder automatically"""
        current_dir = os.getcwd()
        downloads_path = os.path.expanduser("~/Downloads")
        search_paths = [current_dir, downloads_path]
        
        print(f"🔍 Searching for Vosk model...")
        
        for search_dir in search_paths:
            if not os.path.exists(search_dir):
                continue
                
            for item in os.listdir(search_dir):
                if 'vosk' in item.lower() and os.path.isdir(os.path.join(search_dir, item)):
                    model_path = os.path.join(search_dir, item)
                    
                    # Verify it's a valid Vosk model
                    required_dirs = ['am', 'graph']
                    if all(os.path.exists(os.path.join(model_path, d)) for d in required_dirs):
                        print(f"✅ Found model: {model_path}")
                        return model_path
        
        raise FileNotFoundError("Vosk model not found. Please ensure the model folder is in your project directory or Downloads folder.")

    def start_transcription(self):
        """Start real-time audio transcription"""
        self.is_listening = True
        print("🎤 Starting microphone... Please wait...")
        
        def audio_callback(indata, frames, time, status):
            if status and "overflow" not in str(status).lower():
                print(f"⚠️  Audio status: {status}")
            
            if not self.is_listening or self.is_processing:
                return
            
            try:
                # Convert audio data to bytes
                audio_data = bytes(indata)
                
                if self.rec.AcceptWaveform(audio_data):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "").strip()
                    
                    if text and len(text) > 2:  # Only process meaningful text
                        print(f"\n🎯 Transcribed: {text}")
                        # Process in separate thread to avoid blocking
                        threading.Thread(target=self.generate_ai_response, args=(text,), daemon=True).start()
                
            except Exception as e:
                if "overflow" not in str(e).lower():
                    print(f"❌ Audio processing error: {e}")

        try:
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=4000,
                dtype='int16',
                channels=1,
                callback=audio_callback,
                latency='low'
            ):
                print("🎤 Listening... (Press Ctrl+C to stop)")
                print("💬 You can speak in Arabic now!")
                
                while self.is_listening:
                    sd.sleep(100)
                    
        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
            self.stop_transcription()
        except Exception as e:
            print(f"❌ Error in transcription: {e}")
            print("🔧 Try running as Administrator or check microphone permissions")

    def stop_transcription(self):
        """Stop the transcription process"""
        self.is_listening = False
        print("✅ Transcription stopped")

    def generate_ai_response(self, text):
        """Generate AI response from transcribed text"""
        if self.is_processing:
            return
            
        self.is_processing = True
        
        try:
            print(f"👤 Patient: {text}")
            
            # Add user message to conversation history
            self.full_transcript.append({"role": "user", "content": text})
            
            print("🤖 Generating response...")
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="google/gemma-2-9b-it",
                messages=self.full_transcript,
                max_tokens=150,
                temperature=0.7
            )

            ai_response = response.choices[0].message.content.strip()
            self.generate_audio(ai_response)
            
        except Exception as e:
            print(f"❌ Error generating AI response: {e}")
            error_response = "I apologize, I'm having technical difficulties. Could you please repeat that?"
            self.generate_audio(error_response)
        
        finally:
            self.is_processing = False

    def generate_audio(self, text):
        """Generate and play audio response"""
        try:
            # Add assistant message to conversation history
            self.full_transcript.append({"role": "assistant", "content": text})
            print(f"🤖 AI Receptionist: {text}")

            # Generate audio using gTTS
            tts = gTTS(text=text, lang="ar", slow=False)  # Arabic TTS
            filename = "ai_reply.mp3"
            tts.save(filename)
            
            # Play audio based on operating system
            import platform
            system = platform.system().lower()
            
            print("🔊 Playing response...")
            if system == "windows":
                os.system(f'start "" "{filename}"')
            elif system == "darwin":  # macOS
                os.system(f"afplay {filename}")
            elif system == "linux":
                os.system(f"mpg123 {filename}")
            
            # Small delay to ensure audio starts playing
            time.sleep(0.5)
                
        except Exception as e:
            print(f"❌ Error generating audio: {e}")
            print("🔧 Make sure you have internet connection for TTS")

    def test_microphone(self):
        """Test microphone before starting"""
        print("🎤 Testing microphone...")
        
        try:
            # Record 2 seconds to test
            test_audio = sd.rec(int(16000 * 2), samplerate=16000, channels=1, dtype='int16')
            print("🔴 Recording test... speak now for 2 seconds!")
            sd.wait()
            
            volume = np.abs(test_audio).mean()
            print(f"📊 Audio level: {volume}")
            
            if volume > 100:
                print("✅ Microphone is working!")
                return True
            else:
                print("❌ Low/no audio detected")
                print("🔧 Check microphone permissions and volume")
                return False
                
        except Exception as e:
            print(f"❌ Microphone test failed: {e}")
            return False

    def run(self):
        """Main method to run the assistant"""
        print("=" * 50)
        print("🏥 Vancouver Dental Clinic - AI Receptionist")
        print("=" * 50)
        
        # Test microphone first
        if not self.test_microphone():
            print("\n⚠️  Microphone issues detected. Continue anyway? (y/n)")
            if input().lower() != 'y':
                return
        
        # Initial greeting
        greeting = "مرحباً، شكراً لاتصالكم بعيادة فانكوفر لطب الأسنان. اسمي ساندي، كيف يمكنني مساعدتكم؟"
        print("\n🤖 Starting AI Assistant...")
        time.sleep(1)
        self.generate_audio(greeting)
        
        # Wait for greeting to finish
        time.sleep(3)
        
        # Start transcription
        try:
            self.start_transcription()
        except KeyboardInterrupt:
            print("\n👋 Thank you for using AI Assistant!")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")

# Additional utility functions
def check_requirements():
    """Check if all required packages are installed"""
    required_packages = ['vosk', 'sounddevice', 'openai', 'gtts', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print(f"📦 Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main function to run the assistant"""
    print("🎯 AI Voice Assistant - Dental Clinic")
    print("Built with Vosk (Arabic) + OpenAI + gTTS")
    
    # Check requirements
    if not check_requirements():
        return
    
    try:
        # Initialize and run assistant
        ai_assistant = AI_Assistant()
        ai_assistant.run()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("📥 Download the Arabic model from: https://alphacephei.com/vosk/models/")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("🔧 Try running as Administrator or check your setup")

if __name__ == "__main__":
    main()