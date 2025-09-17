"""
Arabic Voice Assistant - Online Deployment Version
Optimized for Streamlit Cloud, Hugging Face Spaces, and Google Colab
"""

import streamlit as st
import numpy as np
import tempfile
import os
import base64
import json
import time
import io
from pathlib import Path
import requests

# Core imports with fallbacks
try:
    from gtts import gTTS
    from openai import OpenAI
    import speech_recognition as sr
    from pydub import AudioSegment
except ImportError as e:
    st.error(f"Missing package: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🦷 عيادة فانكوفر لطب الأسنان",
    page_icon="🦷",
    layout="wide"
)

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'assistant_ready' not in st.session_state:
    st.session_state.assistant_ready = True  # Always ready for online version
if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = "hybrid"  # "text", "audio", "hybrid"
if 'first_message_sent' not in st.session_state:
    st.session_state.first_message_sent = False

class OnlineArabicVoiceAssistant:
    def __init__(self):
        """Initialize Online Arabic Voice Assistant"""
        # Check if running in different environments
        self.is_colab = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
        self.is_local = not (os.environ.get('STREAMLIT_SHARING_MODE') or 
                           os.environ.get('HUGGINGFACE_HUB_CACHE'))
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )
        
        # Initialize Speech Recognition
        self.recognizer = sr.Recognizer()
        
        # Arabic system prompt
        self.system_prompt = """أنت ساندي، موظفة استقبال في عيادة فانكوفر لطب الأسنان.

المعلومات المهمة:
- أوقات العمل: الاثنين إلى الجمعة من 8 صباحاً إلى 6 مساءً، السبت من 9 صباحاً إلى 3 مساءً
- الموقع: وسط مدينة فانكوفر
- الهاتف: (604) 555-DENTAL
- الخدمات: طب الأسنان العام، تنظيف الأسنان، الحشوات، التيجان، علاج الجذور، طب الأسنان التجميلي

تعليمات مهمة:
1. اجيبي باللغة العربية فقط
2. كوني ودودة ومهنية
3. اجعلي الردود قصيرة وواضحة
4. اسألي دائماً كيف يمكنك المساعدة أكثر
5. عند حجز المواعيد، اطلبي اسم المريض ونوع الخدمة المطلوبة

مثال للترحيب: "مرحباً، أهلاً وسهلاً بك في عيادة فانكوفر لطب الأسنان. اسمي ساندي، كيف يمكنني مساعدتك اليوم؟"
"""

    def transcribe_audio_google(self, audio_data, language='ar-SA'):
        """Transcribe audio using Google Speech Recognition"""
        try:
            if isinstance(audio_data, str) and audio_data.startswith('data:'):
                # Handle base64 audio data
                header, encoded = audio_data.split(',', 1)
                audio_bytes = base64.b64decode(encoded)
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            elif isinstance(audio_data, bytes):
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            else:
                # Handle file path
                audio_segment = AudioSegment.from_file(audio_data)
            
            # Convert to WAV format for recognition
            wav_data = audio_segment.export(format="wav").read()
            audio_source = sr.AudioFile(io.BytesIO(wav_data))
            
            with audio_source as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio_recorded = self.recognizer.record(source)
            
            # Recognize speech
            text = self.recognizer.recognize_google(
                audio_recorded, 
                language=language
            )
            return text.strip()
            
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            st.error(f"خطأ في خدمة التعرف على الكلام: {e}")
            return ""
        except Exception as e:
            st.error(f"خطأ في معالجة الصوت: {e}")
            return ""

    def generate_response(self, user_text):
        """Generate Arabic response using AI"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add recent chat history
            for msg in st.session_state.chat_history[-5:]:
                if msg["user"]:
                    messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["assistant"]})
            
            # Add current user message
            messages.append({"role": "user", "content": user_text})
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="google/gemma-2-9b-it",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"عذراً، حدث خطأ تقني: {str(e)[:50]}... يرجى المحاولة مرة أخرى."

    def generate_greeting(self):
        """Generate greeting message"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.append({"role": "user", "content": "مرحباً، أود التحدث مع موظفة الاستقبال"})
            
            response = self.openai_client.chat.completions.create(
                model="google/gemma-2-9b-it",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return "مرحباً، أهلاً وسهلاً بك في عيادة فانكوفر لطب الأسنان. اسمي ساندي، كيف يمكنني مساعدتك اليوم؟"

    def text_to_speech(self, text):
        """Convert text to speech using gTTS"""
        try:
            tts = gTTS(text=text, lang='ar', slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            return temp_file.name
        except Exception as e:
            st.error(f"خطأ في تحويل النص إلى صوت: {e}")
            return None

# Initialize assistant
@st.cache_resource
def get_assistant():
    return OnlineArabicVoiceAssistant()

def create_audio_recorder_html():
    """Create HTML5 audio recorder"""
    html_code = """
    <div style="text-align: center; padding: 20px; border: 2px dashed #ccc; border-radius: 10px;">
        <h3>🎙️ مسجل الصوت</h3>
        <div id="recorder-status">اضغط على "ابدأ التسجيل" للبدء</div>
        <br>
        <button id="start-btn" onclick="startRecording()" 
                style="background-color: #ff4444; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px;">
            🎙️ ابدأ التسجيل
        </button>
        <button id="stop-btn" onclick="stopRecording()" disabled
                style="background-color: #888; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px;">
            ⏹️ إيقاف التسجيل
        </button>
        <button id="process-btn" onclick="processRecording()" disabled
                style="background-color: #4444ff; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px;">
            🎯 معالجة التسجيل
        </button>
        <br><br>
        <audio id="audio-playback" controls style="display:none; width: 100%;"></audio>
        <div id="countdown" style="font-size: 24px; color: red; display: none;"></div>
    </div>
    
    <script>
    let mediaRecorder;
    let audioChunks = [];
    let recordingTimer;
    let audioBlob;
    
    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = () => {
                audioBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
                const audioUrl = URL.createObjectURL(audioBlob);
                
                const audioElement = document.getElementById('audio-playback');
                audioElement.src = audioUrl;
                audioElement.style.display = 'block';
                
                document.getElementById('process-btn').disabled = false;
                document.getElementById('recorder-status').textContent = 'تم التسجيل! يمكنك الآن معالجته أو إعادة التسجيل';
            };
            
            mediaRecorder.start();
            audioChunks = [];
            
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('process-btn').disabled = true;
            document.getElementById('recorder-status').textContent = 'جاري التسجيل... اضغط على "إيقاف" عند الانتهاء';
            
            // Auto stop after 10 seconds
            setTimeout(() => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    stopRecording();
                }
            }, 10000);
            
        } catch (error) {
            document.getElementById('recorder-status').textContent = 'خطأ: لا يمكن الوصول للمايك. تأكد من السماح للموقع بالوصول للمايك.';
        }
    }
    
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
    }
    
    function processRecording() {
        if (audioBlob) {
            const reader = new FileReader();
            reader.onloadend = () => {
                // Send audio data to Streamlit
                window.parent.postMessage({
                    type: 'audio-data',
                    data: reader.result
                }, '*');
                
                document.getElementById('recorder-status').textContent = 'تم إرسال التسجيل للمعالجة...';
            };
            reader.readAsDataURL(audioBlob);
        }
    }
    </script>
    """
    return html_code

def process_user_input(assistant, text, mode="text"):
    """Process user input and generate response"""
    if not text or not text.strip():
        st.warning("⚠️ لم يتم إدخال أي نص")
        return
    
    st.success(f"✅ **النص المستلم:** {text}")
    
    # Generate AI response
    with st.spinner("🤖 جاري إنشاء الرد..."):
        response = assistant.generate_response(text.strip())
    
    # Add to chat history
    st.session_state.chat_history.append({
        "user": text.strip(),
        "assistant": response,
        "mode": mode
    })
    
    # Generate audio response if needed
    if st.session_state.chat_mode in ["audio", "hybrid"]:
        with st.spinner("🔊 جاري تحويل الرد إلى صوت..."):
            audio_file = assistant.text_to_speech(response)
            
            if audio_file:
                # Store audio file reference
                msg_index = len(st.session_state.chat_history) - 1
                st.session_state[f"audio_{msg_index}"] = audio_file
    
    st.rerun()

def display_chat_history():
    """Display chat history with audio playback"""
    if not st.session_state.chat_history:
        st.info("💬 ابدأ محادثتك...")
        return
    
    for i, msg in enumerate(st.session_state.chat_history):
        # User message
        if msg["user"]:
            mode_emoji = "🎤" if msg.get("mode") == "audio" else "⌨️"
            st.markdown(f"**👤 المريض {mode_emoji}:** {msg['user']}")
        
        # Assistant response
        st.markdown(f"**🤖 ساندي:** {msg['assistant']}")
        
        # Audio playback if available
        audio_key = f"audio_{i}"
        if audio_key in st.session_state and os.path.exists(st.session_state[audio_key]):
            with open(st.session_state[audio_key], "rb") as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
        
        st.divider()

def send_initial_greeting(assistant):
    """Send initial greeting from assistant"""
    if not st.session_state.first_message_sent:
        with st.spinner("🤖 مرحباً..."):
            greeting = assistant.generate_greeting()
            
            st.session_state.chat_history.append({
                "user": "",
                "assistant": greeting,
                "mode": "system"
            })
            
            # Generate audio greeting if needed
            if st.session_state.chat_mode in ["audio", "hybrid"]:
                audio_file = assistant.text_to_speech(greeting)
                if audio_file:
                    st.session_state["audio_0"] = audio_file
        
        st.session_state.first_message_sent = True
        st.rerun()

def main():
    """Main application"""
    
    # Title and header
    st.title("🦷 عيادة فانكوفر لطب الأسنان")
    st.markdown("### 🤖 مساعدة الاستقبال الذكية - ساندي (النسخة الأونلاين)")
    st.markdown("**مرحباً! يمكنك التحدث باللغة العربية نصياً أو صوتياً**")
    
    # Initialize assistant
    assistant = get_assistant()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ إعدادات النظام")
        
        # System status
        st.success("✅ النظام جاهز للعمل أونلاين")
        
        # Environment info
        if assistant.is_colab:
            st.info("🔬 تم اكتشاف بيئة Google Colab")
        elif not assistant.is_local:
            st.info("☁️ يعمل في بيئة السحابة")
        else:
            st.info("💻 يعمل محلياً")
        
        st.divider()
        
        # Chat mode selection
        st.header("💬 نمط المحادثة")
        chat_mode = st.selectbox(
            "اختر نمط التفاعل:",
            ["نصي", "صوتي", "مختلط"],
            index=2,
            help="النصي: كتابة فقط، الصوتي: تسجيل فقط، المختلط: كلاهما"
        )
        
        # Map selection to internal mode
        mode_map = {"نصي": "text", "صوتي": "audio", "مختلط": "hybrid"}
        st.session_state.chat_mode = mode_map[chat_mode]
        
        st.divider()
        
        # Clear chat
        if st.button("🗑️ مسح المحادثة"):
            st
