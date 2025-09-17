"""
Arabic Voice Assistant - Streamlit Web Version
باستخدام الميكروفون من المتصفح (streamlit-webrtc)
"""

import streamlit as st
import numpy as np
import wave
import tempfile
import os
import base64
import json
import time

from pathlib import Path
from vosk import Model, KaldiRecognizer
from gtts import gTTS
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# ---------------- إعداد الصفحة ----------------
st.set_page_config(
    page_title="🦷 عيادة فانكوفر لطب الأسنان",
    page_icon="🦷",
    layout="wide"
)

# ---------------- الحالة ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "assistant_ready" not in st.session_state:
    st.session_state.assistant_ready = False
if "first_message_sent" not in st.session_state:
    st.session_state.first_message_sent = False


# ---------------- المساعد ----------------
class ArabicVoiceAssistant:
    def __init__(self):
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )
        self.model = None
        self.rec = None
        self.sample_rate = 16000

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
"""

    def find_and_load_model(self):
        try:
            search_paths = [Path.cwd(), Path.home() / "Downloads"]
            for search_dir in search_paths:
                if not search_dir.exists():
                    continue
                for item in search_dir.iterdir():
                    if (
                        item.is_dir()
                        and "vosk" in item.name.lower()
                        and "ar" in item.name.lower()
                    ):
                        if (item / "am").exists() and (item / "graph").exists():
                            self.model = Model(str(item))
                            self.rec = KaldiRecognizer(self.model, self.sample_rate)
                            return True, f"✅ تم تحميل النموذج من: {item.name}"
            return False, "❌ لم يتم العثور على نموذج Vosk العربي"
        except Exception as e:
            return False, f"❌ خطأ في تحميل النموذج: {e}"

    def transcribe_audio_file(self, audio_file_path):
        if not self.rec:
            return ""
        try:
            self.rec = KaldiRecognizer(self.model, self.sample_rate)
            with wave.open(audio_file_path, "rb") as wav_file:
                audio_data = wav_file.readframes(wav_file.getnframes())
                if self.rec.AcceptWaveform(audio_data):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "").strip()
                else:
                    result = json.loads(self.rec.FinalResult())
                    text = result.get("text", "").strip()
                return text
        except Exception as e:
            st.error(f"خطأ في تحويل الصوت إلى نص: {e}")
            return ""

    def generate_response(self, user_text):
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            for msg in st.session_state.chat_history[-5:]:
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["assistant"]})
            messages.append({"role": "user", "content": user_text})

            response = self.openai_client.chat.completions.create(
                model="google/gemma-2-9b-it",
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "عذراً، حدث خطأ تقني. يرجى المحاولة مرة أخرى."

    def text_to_speech(self, text):
        try:
            tts = gTTS(text=text, lang="ar", slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            return temp_file.name
        except Exception as e:
            st.error(f"خطأ في تحويل النص إلى صوت: {e}")
            return None


@st.cache_resource
def get_assistant():
    return ArabicVoiceAssistant()


# ---------------- الصوت من المتصفح ----------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames = []

    def recv_audio(self, frame):
        self.frames.append(frame.to_ndarray())
        return frame

    def save_wav(self, filename="temp.wav"):
        if not self.frames:
            return None
        audio = np.concatenate(self.frames, axis=0).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio.tobytes())
        return filename


# ---------------- واجهة التطبيق ----------------
st.title("🦷 عيادة فانكوفر لطب الأسنان")
st.markdown("### 🤖 ساندي - المساعدة الذكية")
st.info("يمكنك التحدث بالعربية مباشرة من مايكروفون المتصفح 🎤")

assistant = get_assistant()

# تحميل النموذج
if not st.session_state.assistant_ready:
    with st.spinner("🔄 تحميل نموذج Vosk..."):
        success, message = assistant.find_and_load_model()
        if success:
            st.success(message)
            st.session_state.assistant_ready = True
        else:
            st.error(message)
            st.stop()

# تسجيل الصوت من البراوزر
ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if ctx and ctx.audio_processor:
    if st.button("🎯 معالجة التسجيل"):
        wav_file = ctx.audio_processor.save_wav()
        if wav_file:
            with st.spinner("📝 تحويل الصوت إلى نص..."):
                text = assistant.transcribe_audio_file(wav_file)
            if text:
                st.success(f"✅ النص المستخرج: {text}")
                with st.spinner("🤖 توليد الرد..."):
                    reply = assistant.generate_response(text)
                st.session_state.chat_history.append(
                    {"user": text, "assistant": reply}
                )
                st.markdown(f"**👤 المريض:** {text}")
                st.markdown(f"**🤖 ساندي:** {reply}")
                audio_file = assistant.text_to_speech(reply)
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
        else:
            st.warning("⚠️ لم يتم تسجيل أي صوت.")


# ---------------- سجل المحادثة ----------------
st.subheader("📋 سجل المحادثة")
for msg in st.session_state.chat_history:
    if msg["user"]:
        st.markdown(f"**👤 المريض:** {msg['user']}")
    st.markdown(f"**🤖 ساندي:** {msg['assistant']}")
