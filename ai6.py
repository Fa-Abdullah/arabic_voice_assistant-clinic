"""
Fixed Arabic Voice Assistant - Streamlit Interface
With multiple speech recognition options
"""

import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
import base64
from pathlib import Path
import json
import time
import threading
import requests

# Core imports
try:
    from vosk import Model, KaldiRecognizer
    from gtts import gTTS
    from openai import OpenAI
    import whisper
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
    st.session_state.assistant_ready = False
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = "written"
if 'auto_record' not in st.session_state:
    st.session_state.auto_record = False
if 'first_message_sent' not in st.session_state:
    st.session_state.first_message_sent = False
if 'speech_recognition_method' not in st.session_state:
    st.session_state.speech_recognition_method = "openai"  # openai, whisper, or vosk
if 'vosk_model_loaded' not in st.session_state:
    st.session_state.vosk_model_loaded = False
if 'vosk_model' not in st.session_state:
    st.session_state.vosk_model = None

class ArabicVoiceAssistant:
    def __init__(self):
        """Initialize Arabic Voice Assistant"""
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )
        
        self.sample_rate = 16000
        
        # Pure Arabic system prompt
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

    def transcribe_audio_openai(self, audio_file_path):
        """Transcribe audio using OpenAI API"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language="ar"
                )
            return transcript.text
        except Exception as e:
            st.error(f"❌ خطأ في التعرف على الصوت عبر OpenAI: {e}")
            return ""

    def transcribe_audio_whisper(self, audio_file_path):
        """Transcribe audio using local Whisper"""
        try:
            if not hasattr(st.session_state, 'whisper_model') or st.session_state.whisper_model is None:
                with st.spinner("🔄 جاري تحميل نموذج Whisper المحلي..."):
                    st.session_state.whisper_model = whisper.load_model("base")
            
            result = st.session_state.whisper_model.transcribe(audio_file_path, language="ar")
            return result["text"].strip()
        except Exception as e:
            st.error(f"❌ خطأ في التعرف على الصوت عبر Whisper المحلي: {e}")
            return ""

    def transcribe_audio_vosk(self, audio_file_path):
        """Transcribe audio using Vosk"""
        try:
            if not st.session_state.vosk_model_loaded:
                with st.spinner("🔄 جاري تحميل نموذج Vosk..."):
                    # Try to find Vosk model
                    search_paths = [
                        Path.cwd() / "vosk-model-ar",
                        Path.home() / "Downloads" / "vosk-model-ar",
                        Path.home() / "vosk-model-ar"
                    ]
                    
                    model_path = None
                    for path in search_paths:
                        if path.exists():
                            model_path = str(path)
                            break
                    
                    if model_path:
                        st.session_state.vosk_model = Model(model_path)
                        st.session_state.vosk_model_loaded = True
                    else:
                        st.error("❌ لم يتم العثور على نموذج Vosk العربي")
                        return ""
            
            rec = KaldiRecognizer(st.session_state.vosk_model, self.sample_rate)
            
            with wave.open(audio_file_path, 'rb') as wav_file:
                # Check sample rate
                if wav_file.getframerate() != self.sample_rate:
                    st.warning("⚠️ معدل العينة غير متطابق، قد تكون النتائج غير دقيقة")
                
                # Read and process audio data
                while True:
                    data = wav_file.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        pass
                
                result = json.loads(rec.FinalResult())
                return result.get("text", "").strip()
                
        except Exception as e:
            st.error(f"❌ خطأ في التعرف على الصوت عبر Vosk: {e}")
            return ""

    def transcribe_audio_file(self, audio_file_path):
        """Transcribe audio file using the selected method"""
        if not os.path.exists(audio_file_path):
            st.error("❌ الملف الصوتي غير موجود")
            return ""
        
        method = st.session_state.speech_recognition_method
        
        if method == "openai":
            return self.transcribe_audio_openai(audio_file_path)
        elif method == "whisper":
            return self.transcribe_audio_whisper(audio_file_path)
        elif method == "vosk":
            return self.transcribe_audio_vosk(audio_file_path)
        else:
            return ""

    def generate_response(self, user_text):
        """Generate Arabic response"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add recent chat history (last 5 messages)
            for msg in st.session_state.chat_history[-5:]:
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
            return "عذراً، حدث خطأ تقني. يرجى المحاولة مرة أخرى."

    def generate_greeting(self):
        """Generate a greeting message"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.append({"role": "user", "content": "مرحباً"})
            
            response = self.openai_client.chat.completions.create(
                model="google/gemma-2-9b-it",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return "مرحباً، أهلاً وسهلاً بك في عيادة فانكوفر لطب الأسنان. اسمي ساندي، كيف يمكنني مساعدتك اليوم؟"

    def text_to_speech(self, text):
        """Generate Arabic TTS"""
        try:
            tts = gTTS(text=text, lang='ar', slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", delete=False)
            tts.save(temp_file.name)
            return temp_file.name
        except Exception as e:
            st.error(f"خطأ في تحويل النص إلى صوت: {e}")
            return None

# Initialize assistant
@st.cache_resource
def get_assistant():
    return ArabicVoiceAssistant()

def record_audio(duration=5):
    """Record audio for specified duration"""
    try:
        st.info(f"🔴 جاري التسجيل... ({duration} ثواني)")
        
        # Record audio
        audio_data = sd.rec(int(duration * 16000), 
                          samplerate=16000, 
                          channels=1, 
                          dtype='float32')
        
        # Show countdown
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(duration):
            progress_bar.progress((i + 1) / duration)
            status_text.text(f"🎤 التسجيل جاري... {duration - i} ثواني متبقية")
            time.sleep(1)
        
        sd.wait()
        progress_bar.empty()
        status_text.empty()
        
        # Convert to int16 and save as WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())
        
        return temp_file.name
    
    except Exception as e:
        st.error(f"خطأ في التسجيل: {e}")
        return None

def play_audio(audio_file):
    """Play audio in browser"""
    if audio_file and os.path.exists(audio_file):
        try:
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            
            audio_html = f"""
            <audio controls autoplay style="width: 100%;">
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                المتصفح لا يدعم تشغيل الصوت
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ خطأ في تشغيل الصوت: {e}")

def send_greeting(assistant):
    """Send greeting message from assistant"""
    if not st.session_state.first_message_sent:
        with st.spinner("🤖 جاري التحضير..."):
            greeting = assistant.generate_greeting()
            
            st.session_state.chat_history.append({
                "user": "",
                "assistant": greeting
            })
            
            if st.session_state.chat_mode in ["audio", "twins"]:
                with st.spinner("🔊 جاري تحويل التحية إلى صوت..."):
                    audio_file = assistant.text_to_speech(greeting)
                
                if audio_file and os.path.exists(audio_file):
                    st.session_state["audio_greeting"] = audio_file
                    play_audio(audio_file)
        
        st.session_state.first_message_sent = True
        st.rerun()

def main():
    """Main application"""
    
    st.title("🦷 عيادة فانكوفر لطب الأسنان")
    st.markdown("### 🤖 مساعدة الاستقبال الذكية - ساندي")
    st.markdown("**مرحباً وأهلاً بكم! يمكنكم التحدث باللغة العربية**")
    
    assistant = get_assistant()
    
    with st.sidebar:
        st.header("⚙️ إعدادات النظام")
        
        if not st.session_state.assistant_ready:
            with st.spinner("🔄 جاري التحضير..."):
                st.session_state.assistant_ready = True
                st.success("✅ النظام جاهز")
        
        st.divider()
        
        st.header("🎙️ طريقة التعرف على الصوت")
        recognition_method = st.radio(
            "اختر طريقة التعرف على الصوت:",
            ["OpenAI API", "Whisper محلي", "Vosk"],
            captions=["الأفضل (يتطلب اتصال بالإنترنت)", "متوسط الجودة", "بديل احتياطي"],
            index=0
        )
        
        method_map = {
            "OpenAI API": "openai",
            "Whisper محلي": "whisper", 
            "Vosk": "vosk"
        }
        st.session_state.speech_recognition_method = method_map[recognition_method]
        
        st.divider()
        
        st.header("💬 نمط المحادثة")
        chat_mode = st.radio(
            "اختر نمط المحادثة:",
            ["نصي", "صوتي", "توأم"],
            captions=["محادثة نصية عادية", "تسجيل صوتي تلقائي", "نص وصوت معاً"],
            index=0
        )
        
        mode_map = {"نصي": "written", "صوتي": "audio", "توأم": "twins"}
        st.session_state.chat_mode = mode_map[chat_mode]
        
        if st.session_state.chat_mode in ["audio", "twins"]:
            st.subheader("🎤 إعدادات التسجيل")
            record_duration = st.slider("مدة التسجيل (ثانية)", 3, 10, 5)
            
            if st.session_state.chat_mode == "audio":
                st.session_state.auto_record = st.checkbox("التسجيل التلقائي بعد الرد", value=True)
        
        st.divider()
        
        if st.button("🗑️ مسح المحادثة"):
            st.session_state.chat_history = []
            st.session_state.first_message_sent = False
            st.rerun()
        
        st.divider()
        
        st.header("🏥 معلومات العيادة")
        st.info("""
        **عيادة فانكوفر لطب الأسنان**
        
        📍 وسط مدينة فانكوفر  
        📞 (604) 555-DENTAL
        
        **أوقات العمل:**
        • الاثنين-الجمعة: 8ص-6م
        • السبت: 9ص-3م  
        • الأحد: مغلق
        
        **الخدمات:**
        • طب الأسنان العام
        • تنظيف وفحص الأسنان
        • الحشوات والتيجان
        • علاج الجذور
        • طب الأسنان التجميلي
        """)

    if st.session_state.assistant_ready and not st.session_state.first_message_sent:
        send_greeting(assistant)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 المحادثة")
        
        mode_display = {
            "written": "📝 النمط النصي",
            "audio": "🎤 النمط الصوتي", 
            "twins": "👥 النمط التوأم"
        }
        st.info(f"**النمط الحالي:** {mode_display[st.session_state.chat_mode]}")
        
        method_display = {
            "openai": "OpenAI API 🌐",
            "whisper": "Whisper محلي 💻",
            "vosk": "Vosk 🔄"
        }
        st.info(f"**طريقة التعرف على الصوت:** {method_display[st.session_state.speech_recognition_method]}")
        
        if st.session_state.chat_mode in ["audio", "twins"]:
            st.subheader("🎤 الإدخال الصوتي")
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                if st.button("🎙️ ابدأ التسجيل", disabled=not st.session_state.assistant_ready):
                    audio_file = record_audio(record_duration)
                    if audio_file and os.path.exists(audio_file):
                        st.session_state.last_recording = audio_file
                        st.success("✅ تم التسجيل بنجاح!")
                        st.audio(audio_file, format="audio/wav")
                    else:
                        st.error("❌ فشل في التسجيل")
            
            with col_rec2:
                if st.button("🎯 معالجة التسجيل") and hasattr(st.session_state, 'last_recording'):
                    if os.path.exists(st.session_state.last_recording):
                        with st.spinner("🔄 جاري تحويل الصوت إلى نص..."):
                            text = assistant.transcribe_audio_file(st.session_state.last_recording)
                            
                            if text:
                                st.success(f"✅ تم التعرف على النص: {text}")
                                process_user_input(assistant, text)
                            else:
                                st.warning("⚠️ لم يتم التعرف على كلام. حاول مرة أخرى.")
                    else:
                        st.error("❌ ملف التسجيل غير موجود")
            
            st.subheader("📁 رفع ملف صوتي")
            uploaded_file = st.file_uploader("اختر ملف صوتي (WAV, MP3)", type=['wav', 'mp3'], key="audio_uploader")
            
            if uploaded_file and st.button("🎯 معالجة الملف المرفوع", key="process_uploaded"):
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, delete=False)
                temp_file.write(uploaded_file.read())
                temp_file.close()
                
                if os.path.exists(temp_file.name):
                    with st.spinner("🔄 معالجة الملف الصوتي..."):
                        text = assistant.transcribe_audio_file(temp_file.name)
                        
                        if text:
                            st.success(f"✅ تم التعرف على النص: {text}")
                            process_user_input(assistant, text)
                        else:
                            st.warning("⚠️ لم يتم التعرف على كلام في الملف.")
                    
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                else:
                    st.error("❌ فشل في حفظ الملف المرفوع")
        
        if st.session_state.chat_mode in ["written", "twins"]:
            st.subheader("⌨️ الإدخال النصي")
            user_text = st.text_area("اكتب رسالتك هنا:", height=100, 
                                    placeholder="مثال: ما هي مواعيد العيادة؟", key="text_input")
            
            if st.button("📤 إرسال الرسالة", key="send_text") and user_text.strip():
                process_user_input(assistant, user_text.strip())
        
        st.divider()
        st.subheader("📋 سجل المحادثة")
        
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["user"]:
                st.markdown(f"**👤 المريض:** {msg['user']}")
            st.markdown(f"**🤖 ساندي:** {msg['assistant']}")
            
            if st.session_state.chat_mode in ["audio", "twins"]:
                audio_key = f"audio_{i}" if i > 0 else "audio_greeting"
                if audio_key in st.session_state and os.path.exists(st.session_state[audio_key]):
                    st.audio(st.session_state[audio_key], format="audio/mp3")
            
            st.divider()

    with col2:
        st.header("📊 حالة النظام")
        
        st.metric("🤖 حالة المساعد", "جاهز" if st.session_state.assistant_ready else "تحميل")
        st.metric("💬 عدد الرسائل", len(st.session_state.chat_history))
        st.metric("🎯 نمط المحادثة", mode_display[st.session_state.chat_mode])
        st.metric("🎙️ طريقة التعرف", method_display[st.session_state.speech_recognition_method])
        
        if st.session_state.chat_history:
            last_msg = st.session_state.chat_history[-1]
            display_text = last_msg['user'] if last_msg["user"] else last_msg['assistant']
            st.text_area("آخر رسالة:", value=display_text, height=100, disabled=True)
        
        st.divider()
        st.subheader("🎤 اختبار الميكروفون")
        if st.button("اختبار الميكروفون"):
            test_microphone(assistant)

def process_user_input(assistant, text):
    """Process user input and generate response"""
    with st.spinner("🤖 جاري إنشاء الرد..."):
        response = assistant.generate_response(text)
    
    st.session_state.chat_history.append({
        "user": text,
        "assistant": response
    })
    
    if st.session_state.chat_mode in ["audio", "twins"]:
        with st.spinner("🔊 جاري تحويل الرد إلى صوت..."):
            audio_file = assistant.text_to_speech(response)
        
        if audio_file and os.path.exists(audio_file):
            msg_index = len(st.session_state.chat_history) - 1
            st.session_state[f"audio_{msg_index}"] = audio_file
            
            st.success("🔊 تشغيل الرد...")
            play_audio(audio_file)
            
            if st.session_state.chat_mode == "audio" and st.session_state.auto_record:
                st.info("⏳ الانتظار قليلاً قبل التسجيل التلقائي...")
                time.sleep(3)
                st.rerun()
    
    st.rerun()

def test_microphone(assistant):
    """Test microphone"""
    try:
        st.info("🎤 اختبار الميكروفون لمدة 3 ثوان...")
        
        audio_data = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
        
        progress_bar = st.progress(0)
        for i in range(3):
            progress_bar.progress((i + 1) / 3)
            time.sleep(1)
        
        sd.wait()
        progress_bar.empty()
        
        volume = np.abs(audio_data).mean()
        
        if volume > 0.001:
            st.success(f"✅ الميكروفون يعمل بشكل جيد! مستوى الصوت: {volume:.4f}")
            
            # Save test audio
            audio_int16 = (audio_data * 32767).astype(np.int16)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", delete=False)
            temp_file.close()
            
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_int16.tobytes())
            
            if os.path.exists(temp_file.name):
                with st.spinner("🔊 اختبار التعرف على الكلام..."):
                    text = assistant.transcribe_audio_file(temp_file.name)
                    
                    if text.strip():
                        st.success(f"✅ تم التعرف على: '{text}'")
                    else:
                        st.warning("⚠️ لم يتم التعرف على كلام في التسجيل")
                
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
        else:
            st.error("❌ لم يتم اكتشاف صوت. تحقق من إعدادات الميكروفون.")
            
    except Exception as e:
        st.error(f"❌ فشل اختبار الميكروفون: {e}")

if __name__ == "__main__":
    main()