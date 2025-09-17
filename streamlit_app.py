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
  import speech\_recognition as sr
  from pydub import AudioSegment
except ImportError as e:
  st.error(f"Missing package: {e}")
  st.stop()

# Page config

st.set\_page\_config(
  page\_title="🦷 عيادة فانكوفر لطب الأسنان",
  page\_icon="🦷",
  layout="wide"
)

# Session state

if 'chat\_history' not in st.session\_state:
  st.session\_state.chat\_history = \[]
if 'assistant\_ready' not in st.session\_state:
  st.session\_state.assistant\_ready = True
if 'chat\_mode' not in st.session\_state:
  st.session\_state.chat\_mode = "hybrid"  # "text", "audio", "hybrid"
if 'first\_message\_sent' not in st.session\_state:
  st.session\_state.first\_message\_sent = False

# --------- Assistant Class ----------

class OnlineArabicVoiceAssistant:
  def **init**(self):
    self.openai\_client = OpenAI(
      base\_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)",
      api\_key="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"
    )
    self.recognizer = sr.Recognizer()
    self.system\_prompt = """أنت ساندي، موظفة استقبال في عيادة فانكوفر لطب الأسنان.

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

  def transcribe\_audio\_google(self, audio\_data, language='ar-SA'):
    try:
      if isinstance(audio\_data, str) and audio\_data.startswith('data:'):
        header, encoded = audio\_data.split(',', 1)
        audio\_bytes = base64.b64decode(encoded)
        audio\_segment = AudioSegment.from\_file(io.BytesIO(audio\_bytes))
      elif isinstance(audio\_data, bytes):
        audio\_segment = AudioSegment.from\_file(io.BytesIO(audio\_data))
      else:
        audio\_segment = AudioSegment.from\_file(audio\_data)

  ```
        wav_data = audio_segment.export(format="wav").read()
        audio_source = sr.AudioFile(io.BytesIO(wav_data))

        with audio_source as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio_recorded = self.recognizer.record(source)

        text = self.recognizer.recognize_google(audio_recorded, language=language)
        return text.strip()
    except:
        return ""
  ```

  def generate\_response(self, user\_text):
    try:
      messages = \[{"role": "system", "content": self.system\_prompt}]
      for msg in st.session\_state.chat\_history\[-5:]:
      if msg\["user"]:
      messages.append({"role": "user", "content": msg\["user"]})
      messages.append({"role": "assistant", "content": msg\["assistant"]})
      messages.append({"role": "user", "content": user\_text})

    ```
        response = self.openai_client.chat.completions.create(
            model="google/gemma-2-9b-it",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"عذراً، حدث خطأ: {str(e)}"
  ```

  def generate\_greeting(self):
    return "مرحباً، أهلاً وسهلاً بك في عيادة فانكوفر لطب الأسنان. اسمي ساندي، كيف يمكنني مساعدتك اليوم؟"

  def text\_to\_speech(self, text):
    try:
      tts = gTTS(text=text, lang='ar', slow=False)
      temp\_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
      tts.save(temp\_file.name)
      return temp\_file.name
    except:
      return None

  @st.cache\_resource
  def get\_assistant():
    return OnlineArabicVoiceAssistant()

# --------- Helper Functions ----------

  def process\_user\_input(assistant, text, mode="text"):
    if not text.strip():
      return
    response = assistant.generate\_response(text.strip())
    st.session\_state.chat\_history.append({"user": text.strip(), "assistant": response, "mode": mode})
    if st.session\_state.chat\_mode in \["audio", "hybrid"]:
      audio\_file = assistant.text\_to\_speech(response)
    if audio\_file:
      st.session\_state\[f"audio\_{len(st.session\_state.chat\_history)-1}"] = audio\_file
      st.rerun()

  def display\_chat\_history():
    for i, msg in enumerate(st.session\_state.chat\_history):
      if msg\["user"]:
        st.markdown(f"**👤 المريض:** {msg\['user']}")
        st.markdown(f"**🤖 ساندي:** {msg\['assistant']}")
        audio\_key = f"audio\_{i}"
      if audio\_key in st.session\_state and os.path.exists(st.session\_state\[audio\_key]):
        with open(st.session\_state\[audio\_key], "rb") as f:
        st.audio(f.read(), format="audio/mp3")
        st.divider()

  def test\_microphone():
    try:
      st.info("🎤 اختبار الميكروفون لمدة 3 ثوان...")
      audio\_data = sd.rec(int(3\*16000), samplerate=16000, channels=1, dtype='float32')
      sd.wait()
      volume = np.abs(audio\_data).mean()
    if volume > 0.001:
      st.success(f"✅ الميكروفون يعمل! مستوى الصوت: {volume:.4f}")
    else:
st.error("❌ لم يتم اكتشاف صوت.")
except Exception as e:
st.error(f"❌ خطأ في الميكروفون: {e}")

# --------- Main App ----------

def main():
assistant = get\_assistant()

```
col1, col2 = st.columns([2,1])

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
    user_text = st.text_area("✍️ اكتب رسالتك:", height=100, placeholder="مثال: ما هي مواعيد العيادة؟")
    if st.button("📤 إرسال"):
        process_user_input(assistant, user_text, mode="text")

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
```

if **name** == "**main**":
main()
