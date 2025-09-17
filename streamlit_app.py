import streamlit as st
import numpy as np
import tempfile
import os
import base64
import time
import io
import sounddevice as sd
from pathlib import Path

# Core imports
try:
    from gtts import gTTS
    from openai import OpenAI
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.playback import play
except ImportError as e:
    st.error(f"Missing package: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="🦷 عيادة فانكوفر لطب الأسنان",
    page_icon="🦷",
    layout="wide"
)

# Session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'assistant_ready' not in st.session_state:
    st.session_state.assistant_ready = True
if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = "hybrid"  # "text", "audio", "hybrid"
if 'first_message_sent' not in st.session_state:
    st.session_state.first_message_sent = False

# --------- Assistant Class ----------
class OnlineArabicVoiceAssistant:
    def __init__(self):
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )
        self.recognizer = sr.Recognizer()
        self.system_prompt = """أنت ساندي، موظفة استقبال في عيادة فانكوفر لطب الأسنان.

المعلومات المهمة:

* أوقات العمل: الاثنين إلى الجمعة من 8 صباحاً إلى 6 مساءً، السبت من 9 صباحاً إلى 3 مساءً
* الموقع: وسط مدينة فانكوفر
* الهاتف: (604) 555-DENTAL
* الخدمات: طب الأسنان العام، تنظيف الأسنان، الحشوات، التيجان، علاج الجذور، طب الأسنان التجميلي

التعليمات:

* الرد بالعربية فقط
* كوني ودودة ومهنية
* اجعلي الردود قصيرة وواضحة
"""

    def transcribe_audio_google(self, audio_data, language='ar-SA'):
        try:
            if isinstance(audio_data, str) and audio_data.startswith('data:'):
                header, encoded = audio_data.split(',', 1)
                audio_bytes = base64.b64decode(encoded)
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            elif isinstance(audio_data, bytes):
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            else:
                audio_segment = AudioSegment.from_file(audio_data)

            wav_data = audio_segment.export(format="wav").read()
            audio_source = sr.AudioFile(io.BytesIO(wav_data))

            with audio_source as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio_recorded = self.recognizer.record(source)

            text = self.recognizer.recognize_google(audio_recorded, language=language)
            return text.strip()
        except Exception as e:
            st.error(f"خطأ في التعرف على الصوت: {str(e)}")
            return ""

    def generate_response(self, user_text):
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # إضافة تاريخ المحادثة (آخر 5 رسائل)
            for msg in st.session_state.chat_history[-5:]:
                if msg["user"]:
                    messages.append({"role": "user", "content": msg["user"]})
                    messages.append({"role": "assistant", "content": msg["assistant"]})
            
            # إضافة الرسالة الحالية
            messages.append({"role": "user", "content": user_text})

            response = self.openai_client.chat.completions.create(
                model="google/gemma-2-9b-it",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"عذراً، حدث خطأ: {str(e)}"

    def generate_greeting(self):
        return "مرحباً، أهلاً وسهلاً بك في عيادة فانكوفر لطب الأسنان. اسمي ساندي، كيف يمكنني مساعدتك اليوم؟"

    def text_to_speech(self, text):
        try:
            tts = gTTS(text=text, lang='ar', slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            return temp_file.name
        except Exception as e:
            st.error(f"خطأ في تحويل النص إلى كلام: {str(e)}")
            return None

@st.cache_resource
def get_assistant():
    return OnlineArabicVoiceAssistant()

# --------- Helper Functions ----------
def process_user_input(assistant, text, mode="text"):
    if not text.strip():
        return
    
    # إضافة رسالة المستخدم إلى السجل
    st.session_state.chat_history.append({"user": text.strip(), "assistant": "... جاري الرد", "mode": mode})
    
    # توليد الرد
    response = assistant.generate_response(text.strip())
    
    # تحديث الرد في السجل
    st.session_state.chat_history[-1]["assistant"] = response
    
    # تحويل النص إلى كلام إذا كان الوضع صوتي أو مختلط
    if st.session_state.chat_mode in ["audio", "hybrid"]:
        audio_file = assistant.text_to_speech(response)
        if audio_file:
            st.session_state[f"audio_{len(st.session_state.chat_history)-1}"] = audio_file
    
    st.rerun()

def display_chat_history():
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["user"]:
            st.markdown(f"**👤 المريض:** {msg['user']}")
            st.markdown(f"**🤖 ساندي:** {msg['assistant']}")
            
            # عرض الصوت إذا كان متوفراً
            audio_key = f"audio_{i}"
            if audio_key in st.session_state and os.path.exists(st.session_state[audio_key]):
                with open(st.session_state[audio_key], "rb") as f:
                    audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mp3")
            
            st.divider()

def test_microphone():
    try:
        st.info("🎤 اختبار الميكروفون لمدة 3 ثوان...")
        audio_data = sd.rec(int(3*16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        volume = np.abs(audio_data).mean()
        if volume > 0.001:
            st.success(f"✅ الميكروفون يعمل! مستوى الصوت: {volume:.4f}")
        else:
            st.error("❌ لم يتم اكتشاف صوت.")
    except Exception as e:
        st.error(f"❌ خطأ في الميكروفون: {e}")

def record_audio(assistant):
    try:
        st.info("🎙️ تسجيل الصوت... تحدث الآن (5 ثوان)")
        audio_data = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
        sd.wait()
        
        # حفظ الصوت المؤقت
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        import scipy.io.wavfile as wav
        wav.write(temp_audio.name, 16000, audio_data)
        
        # تحويل الصوت إلى نص
        text = assistant.transcribe_audio_google(temp_audio.name)
        
        # تنظيف الملف المؤقت
        os.unlink(temp_audio.name)
        
        return text
    except Exception as e:
        st.error(f"❌ خطأ في التسجيل: {e}")
        return ""

# --------- Main App ----------
def main():
    assistant = get_assistant()
    
    # إضافة رسالة ترحيبية إذا كانت المحادثة فارغة
    if not st.session_state.chat_history:
        greeting = assistant.generate_greeting()
        st.session_state.chat_history.append({"user": "", "assistant": greeting, "mode": "system"})
        
        # تحويل الترحيب إلى صوت
        if st.session_state.chat_mode in ["audio", "hybrid"]:
            audio_file = assistant.text_to_speech(greeting)
            if audio_file:
                st.session_state["audio_0"] = audio_file

    col1, col2 = st.columns([2, 1])

    with col1:
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

        st.header("💬 المحادثة")
        
        # خيارات نمط المحادثة
        chat_mode = st.radio(
            "اختر نمط المحادثة:",
            ["نصي", "صوتي", "مختلط"],
            horizontal=True,
            index=2
        )
        
        # تحويل الخيار إلى قيمة مناسبة
        mode_map = {"نصي": "text", "صوتي": "audio", "مختلط": "hybrid"}
        st.session_state.chat_mode = mode_map[chat_mode]
        
        # واجهة إدخال النص
        user_text = st.text_area("✍️ اكتب رسالتك:", height=100, placeholder="مثال: ما هي مواعيد العيادة؟")
        
        # زر الإرسال للنص
        if st.button("📤 إرسال نص"):
            process_user_input(assistant, user_text, mode="text")
        
        # زر التسجيل الصوتي
        if st.button("🎙️ تسجيل صوتي") and st.session_state.chat_mode in ["audio", "hybrid"]:
            recorded_text = record_audio(assistant)
            if recorded_text:
                process_user_input(assistant, recorded_text, mode="audio")
        
        st.divider()
        st.subheader("📋 سجل المحادثة")
        display_chat_history()

    with col2:
        st.header("📊 حالة النظام")
        st.metric("🤖 نموذج الذكاء الاصطناعي", "جاهز" if st.session_state.assistant_ready else "تحميل")
        st.metric("💬 عدد الرسائل", len(st.session_state.chat_history))
        st.metric("🎯 نمط المحادثة", st.session_state.chat_mode)

        if st.session_state.chat_history:
            last_msg = st.session_state.chat_history[-1]
            if last_msg["user"]:
                st.text_area("آخر رسالة (المريض):", value=last_msg['user'], height=100, disabled=True)
            else:
                st.text_area("آخر رسالة (ساندي):", value=last_msg['assistant'], height=100, disabled=True)

        st.divider()
        st.subheader("🎤 اختبار الميكروفون")
        if st.button("تشغيل الاختبار"):
            test_microphone()
            
        st.divider()
        st.subheader("⚙️ إعدادات")
        if st.button("🗑️ مسح المحادثة"):
            st.session_state.chat_history = []
            # إعادة إضافة رسالة الترحيب
            greeting = assistant.generate_greeting()
            st.session_state.chat_history.append({"user": "", "assistant": greeting, "mode": "system"})
            st.rerun()

if __name__ == "__main__":
    main()
