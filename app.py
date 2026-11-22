import gradio as gr
from ultralytics import YOLO

model = YOLO("model_deteksi_cabai_best.pt")

info_hama = {
    "Sehat": (
        "Daun Sehat",
        (
            "Daun terlihat sehat. Pertahankan penyiraman cukup, pemupukan"
            " berimbang, dan pemantauan rutin."
        ),
    ),
    "Kutu_Kebul": (
        "Terindikasi Kutu Kebul",
        (
            "Pasang perangkap kuning berperekat, bersihkan gulma sekitar,"
            " pangkas daun parah, dan gunakan pestisida nabati/insektisida"
            " sesuai anjuran."
        ),
    ),
    "Thrips": (
        "Terindikasi Hama Thrips",
        (
            "Pangkas bagian daun/pucuk yang keriting parah, pasang perangkap"
            " lekat, dan kendalikan sesuai anjuran."
        ),
    ),
    "Tungau": (
        "Terindikasi Hama Tungau",
        (
            "Semprot bagian bawah daun dengan air bersih, jaga kelembapan,"
            " dan gunakan akarisida/bahan nabati bila diperlukan."
        ),
    ),
}


def prediksi_daun_cabai(gambar):
  if gambar is None:
    return "Silakan unggah foto daun cabai terlebih dahulu.", "-"

  hasil = model.predict(gambar, verbose=False)[0]
  idx = int(hasil.probs.top1)
  kelas = model.names[idx]
  yakin = float(hasil.probs.top1conf) * 100

  label_status, saran = info_hama.get(
      kelas,
      (
          kelas,
          "Terdeteksi kondisi pada daun, lakukan pengecekan lanjutan.",
      ),
  )

  ringkasan = (
      f"### Hasil Prediksi: {kelas}\n"
      f"**Status:** {label_status}\n\n"
      f"**Tingkat Keyakinan:** {yakin:.2f}%"
  )
  return ringkasan, saran


with gr.Blocks() as app:
  gr.Markdown("# Sistem Deteksi Kondisi Daun Cabai")
  gr.Markdown(
      "### Mengapa Menggunakan Klasifikasi?\n"
      "Pendekatan **klasifikasi citra** (`YOLOv8-cls`) dipilih agar sistem"
      " dapat mengenali kondisi keseluruhan foto daun cabai (`Sehat`,"
      " `Kutu_Kebul`, `Thrips`, `Tungau`) secara cepat dan praktis tanpa"
      " mengharuskan pemberian kotak label manual pada tiap objek di"
      " gambar.\n\n"
      "**Petunjuk:** Unggah **foto daun cabai** pada kolom input di kiri,"
      " lalu klik **Deteksi** untuk melihat hasil dan saran penanganan."
  )

  with gr.Row():
    with gr.Column():
      input_gambar = gr.Image(type="pil", label="Input Foto Daun Cabai")
      tombol_deteksi = gr.Button("Deteksi", variant="primary")

    with gr.Column():
      output_hasil = gr.Markdown(label="Hasil Deteksi")
      output_saran = gr.Textbox(
          label="Saran Penanganan Awal", lines=4, interactive=False
      )

  tombol_deteksi.click(
      fn=prediksi_daun_cabai,
      inputs=input_gambar,
      outputs=[output_hasil, output_saran],
  )

if __name__ == "__main__":
  app.launch(theme=gr.themes.Soft())
