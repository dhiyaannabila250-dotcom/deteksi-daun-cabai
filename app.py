import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Deteksi Penyakit & Hama Daun Cabai",
    page_icon="🌶️",
    layout="centered",
)


@st.cache_resource
def load_model():
  return YOLO("model_deteksi_cabai_best.pt")


model = load_model()

info_hama = {
    "Sehat": (
        "Daun Sehat",
        "Daun terlihat sehat. Pertahankan penyiraman cukup, pemupukan berimbang, dan pemantauan rutin.",
    ),
    "Kutu_Kebul": (
        "Terindikasi Kutu Kebul",
        "Pasang perangkap kuning berperekat, bersihkan gulma sekitar, pangkas daun parah, dan gunakan pestisida nabati/insektisida sesuai anjuran.",
    ),
    "Thrips": (
        "Terindikasi Thrips",
        "Periksa bagian bawah daun, jaga kelembapan/irigasi tanaman, singkirkan bagian yang terinfeksi berat, dan aplikasikan pengendalian hama sesuai rekomendasi.",
    ),
    "Tungau": (
        "Terindikasi Tungau",
        "Kurangi kondisi terlalu kering/berdebu di sekitar tanaman, isolasi atau pangkas daun terserang berat, dan gunakan akarisida/pengendalian yang sesuai.",
    ),
}

st.title("🌶️ Klasifikasi & Deteksi Daun Cabai")
st.write(
    "Unggah foto daun cabai untuk mendeteksi kondisi tanaman menggunakan model"
    " YOLOv8."
)

uploaded_file = st.file_uploader(
    "Pilih gambar daun cabai",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
  st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

  if st.button("Deteksi Sekarang", type="primary"):
    with st.spinner("Sedang menganalisis gambar..."):
      results = model(image)
      result = results[0]

      # Tampilkan gambar hasil deteksi (jika ada bounding box)
      plotted = result.plot()
      plotted_rgb = Image.fromarray(plotted[..., ::-1])
      st.image(
          plotted_rgb, caption="Hasil Deteksi", use_container_width=True
      )

      # Cek hasil deteksi objek (boxes) maupun klasifikasi (probs)
      detected_labels = []
      if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
          cls_id = int(box.cls[0].item())
          conf = float(box.conf[0].item())
          label = model.names.get(cls_id, str(cls_id))
          detected_labels.append((label, conf))
      elif result.probs is not None:
        cls_id = int(result.probs.top1)
        conf = float(result.probs.top1conf.item())
        label = model.names.get(cls_id, str(cls_id))
        detected_labels.append((label, conf))

      if detected_labels:
        st.subheader("Hasil Analisis")
        for label, conf in detected_labels:
          judul, saran = info_hama.get(
              label,
              (
                  f"Terdeteksi: {label}",
                  "Lakukan pemeriksaan lanjutan pada tanaman dan sesuaikan penanganan dengan kondisi lapangan.",
              ),
          )
          st.markdown(f"**{judul}** (Keyakinan: `{conf:.2%}`)")
          st.info(saran)
      else:
        st.warning(
            "Tidak ada objek daun/hama yang terdeteksi dengan jelas pada gambar"
            " ini."
        )
