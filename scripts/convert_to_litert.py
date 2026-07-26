"""
Adım 2: ONNX → LiteRT (.tflite)

onnx2tf kütüphanesi ONNX grafını TensorFlow SavedModel'e,
oradan da TFLite / LiteRT formatına dönüştürür.

Çalıştır:
    uv run python scripts/convert_to_litert.py
"""

import os
import numpy as np
import onnx
import onnx2tf

ONNX_PATH   = "outputs/converted/model.onnx"
OUTPUT_DIR  = "outputs/converted/litert_saved"
TFLITE_OUT  = "outputs/converted/model.tflite"


def main():
    # ONNX modelini doğrula
    model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model)
    print(f"✓ ONNX doğrulandı: {ONNX_PATH}")

    # ONNX → TF SavedModel → TFLite (onnx2tf tek adımda halleder)
    onnx2tf.convert(
        input_onnx_file_path=ONNX_PATH,
        output_folder_path=OUTPUT_DIR,
        output_tfv1_pb=False,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )

    # onnx2tf'nin ürettiği .tflite dosyasını bul ve taşı
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".tflite"):
            src = os.path.join(OUTPUT_DIR, f)
            os.rename(src, TFLITE_OUT)
            break

    size_mb = os.path.getsize(TFLITE_OUT) / 1e6
    print(f"✓ LiteRT kaydedildi: {TFLITE_OUT}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()