# streamlit_webrtc_vosk_app.py
# تطبيق Streamlit لتسجيل الميكروفون من المتصفح ومعالجة التحويل إلى نص باستخدام Vosk
# ملاحظات: يعمل مع streamlit-webrtc. لاستخدام هذا التطبيق تحتاج إلى وجود نموذج Vosk محلياً
# أو رفعه يدوياً (أو رابط تحميل) لأن رفعه إلى GitHub غالباً ما يكون كبير الحجم.

import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
import tempfile
import wave
import os
import json
from pathlib import Path
from vosk import Model, KaldiRecognizer
import threading
import zipfile
import requests

# ----------------------- إعداد الصفحة -----------------------
st.set_page_config(page_title="ساندي - استقبال صوتي", layout="wide")
st.title("🦷 ساندي — استقبال صوتي (Streamlit + WebRTC)")
st.markdown("تسجيل الصوت من المتصفح وتحويله إلى نص باستخدام Vosk.")

# ----------------------- إعداد مسارات النموذج -----------------------
MODEL_DIR = Path("./vosk-model-ar-mgb2")  # غيّري هذا المسار لو اسم المجلد مختلف
MODEL_ZIP_NAME = "vosk-model-ar-mgb2.zip"

# ----------------------- وظائف مساعدة -----------------------

def load_vosk_model(model_path: Path):
    """حاول تحميل نموذج Vosk من المسار المحدد"""
    try:
        if not model_path.exists():
            return None, "الموديل غير موجود"
        model = Model(str(model_path))
        return model, None
    except Exception as e:
        return None, str(e)


def download_and_extract_zip(url: str, extract_to: Path):
    """حمّل zip من URL وفك ضغطه إلى المسار المحدد."""
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(MODEL_ZIP_NAME, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        # فك الضغط
        with zipfile.ZipFile(MODEL_ZIP_NAME, 'r') as zf:
            zf.extractall(path=str(extract_to.parent))
        return True, None
    except Exception as e:
        return False, str(e)

# ----------------------- واجهة التحكم -----------------------
st.sidebar.header("إعدادات النموذج")
model_status = st.sidebar.empty()

# خيارات لإضافة النموذج
st.sidebar.write("إذا لم يكن النموذج موجودًا يمكنك: \n - رفع ملف zip للموديل عبر uploader  - أو إعطاء رابط تحميل مباشر (HTTP/HTTPS) ليتم تنزيله هنا")

uploaded_model = st.sidebar.file_uploader("رفع ملف موديل zip (اختياري)", type=["zip"]) 
download_url = st.sidebar.text_input("أو ضع رابط تحميل مباشر للموديل (اختياري)")

# زر لاختبار جاهزية الموديل
if st.sidebar.button("تحديث حالة الموديل"):
    st.experimental_rerun()

# Process uploaded zip
if uploaded_model is not None:
    with open(MODEL_ZIP_NAME, 'wb') as f:
        f.write(uploaded_model.read())
    st.sidebar.success("تم رفع zip. جاري فك الضغط...")
    try:
        with zipfile.ZipFile(MODEL_ZIP_NAME, 'r') as zf:
            zf.extractall(path='.')
        st.sidebar.success("تم فك ضغط الموديل. تأكدي من اسم المجلد.")
    except Exception as e:
        st.sidebar.error(f"فشل فك الضغط: {e}")

# Download from URL if provided
if download_url:
    st.sidebar.info("بدء التحميل من الرابط...")
    ok, err = download_and_extract_zip(download_url, MODEL_DIR)
    if ok:
        st.sidebar.success("اكتملت عملية التحميل والفك.")
    else:
        st.sidebar.error(f"فشل التحميل: {err}")

# Attempt to load model
model, err = load_vosk_model(MODEL_DIR)
if model:
    model_status.info(f"✅ تم تحميل النموذج من: {MODEL_DIR}")
else:
    model_status.warning("⚠️ لم يتم العثور على نموذج Vosk. ضع الموديل بجانب المشروع أو ارفعه كـ zip.")
    if err:
        st.sidebar.error(f"خطأ عند تحميل الموديل: {err}")

# ----------------------- واجهة التسجيل والتحويل -----------------------
st.header("🎤 تسجيل وتشغيل من المتصفح")
st.markdown("اضغطي على زر السماح في المتصفح للسماح باستخدام الميكروفون. يمكنك إيقاف وتشغيل التسجيل بحرية.")

# متغيرات حالة مشتركة
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []
if 'last_wav' not in st.session_state:
    st.session_state.last_wav = None

# مكوّن لمعالجة الصوت الوارد
class RecorderProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = []
        self.lock = threading.Lock()

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # نحصل على مصفوفة numpy من الإطار (float32)
        arr = frame.to_ndarray()
        # arr shape: (channels, samples) أو (samples,) حسب الإطار
        # سنحوّلها إلى 16-bit PCM mono
        if arr.ndim > 1:
            arr_mono = np.mean(arr, axis=0)
        else:
            arr_mono = arr
        # Normalize float32 [-1,1] -> int16
        int16 = (arr_mono * 32767).astype(np.int16)
        with self.lock:
            self.buffer.append(int16.tobytes())
        return frame

    def dump_wav(self, filename, samplerate=48000):
        # streamlit-webrtc يستخدم غالبًا 48000Hz
        with self.lock:
            data = b"".join(self.buffer)
            self.buffer = []
        if not data:
            return False
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(data)
        return True

# بداية WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    audio_processor_factory=RecorderProcessor,
)

# أزرار حفظ ومحو
col1, col2 = st.columns([1,1])
with col1:
    if st.button("🔴 احفظ آخر تسجيل WAV"):
        if webrtc_ctx.audio_processor:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            ok = webrtc_ctx.audio_processor.dump_wav(tmp.name, samplerate=48000)
            if ok:
                st.success(f"✅ تم حفظ التسجيل في: {tmp.name}")
                st.session_state.last_wav = tmp.name
            else:
                st.warning("⚠️ لا يوجد بيانات صوتية لحفظها.")
        else:
            st.error("⚠️ لم يتم تهيئة معالج الصوت بعد.")

with col2:
    if st.button("🗑️ أزل آخر تسجيل محفوظ"):
        if st.session_state.last_wav and os.path.exists(st.session_state.last_wav):
            os.unlink(st.session_state.last_wav)
        st.session_state.last_wav = None
        st.success("تم المسح.")

st.markdown("---")

# رفع ملف WAV بدلا من التسجيل
uploaded_wav = st.file_uploader("أو ارفع ملف WAV جاهز للمعالجة", type=["wav"]) 
if uploaded_wav is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(uploaded_wav.read())
    tmp.close()
    st.session_state.last_wav = tmp.name
    st.success("تم رفع الملف.")

# زر تشغيل التحويل إلى نص
if st.button("🔎 تحويل آخر ملف إلى نص"):
    if not st.session_state.last_wav:
        st.warning("لم يتم تحديد ملف صوتي. سجّلي أو ارفعي ملف أولاً.")
    else:
        wav_path = st.session_state.last_wav
        st.info(f"جارٍ معالجة: {wav_path}")
        if model is None:
            st.error("لا يوجد نموذج Vosk محمّل. يرجى رفع الموديل أو تنزيله أولاً.")
        else:
            try:
                # افتح WAV وتحقق من معدل العينة
                with wave.open(wav_path, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                # لو معدل العينة ليس 16000، نحتاج لتحويل. Vosk يفضل 16000.
                if sample_rate != 16000:
                    st.info(f"معدل العينة {sample_rate}Hz — سيجري إعادة تقسيم إلى 16000Hz.")
                    # استخدم إعادة عيّنة بسيطة: تحويل عبر numpy (قد يكون أسوأ جودة)
                    import audioop
                    frames = audioop.ratecv(frames, 2, 1, sample_rate, 16000, None)[0]
                    sample_rate = 16000
                rec = KaldiRecognizer(model, sample_rate)
                rec.SetWords(False)
                # Feed data in chunks
                CHUNK = 4000
                i = 0
                text_parts = []
                for start in range(0, len(frames), CHUNK):
                    chunk = frames[start:start+CHUNK]
                    if rec.AcceptWaveform(chunk):
                        res = json.loads(rec.Result())
                        text_parts.append(res.get('text', ''))
                final = json.loads(rec.FinalResult())
                text_parts.append(final.get('text', ''))
                transcript = ' '.join([p for p in text_parts if p])
                if transcript.strip() == '':
                    st.warning('لم يتم التعرف على كلام واضح في التسجيل.')
                else:
                    st.session_state.transcriptions.append(transcript)
                    st.success('✅ تم تحويل الصوت إلى نص:')
                    st.write(transcript)
            except Exception as e:
                st.error(f"خطأ أثناء التحويل: {e}")

# عرض السجل
st.markdown("---")
st.subheader("📋 سجل النصوص المحوّلة")
for i, t in enumerate(reversed(st.session_state.transcriptions)):
    st.write(f"{len(st.session_state.transcriptions)-i}. {t}")

st.markdown("---")
st.caption("ملاحظات: \n- Streamlit Cloud لا يدعم الوصول المباشر لهاردوير الميكروفون على الخادم؛ التسجيل يتم من متصفح المستخدم.\n- تأكد من وجود موديل Vosk في مجلد المشروع أو استخدم زر رفع zip/رابط التحميل في الشريط الجانبي.")
