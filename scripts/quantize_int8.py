import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

ONNX_FP32 = "outputs/converted/model.onnx"
ONNX_INT8  = "outputs/converted/model_int8.onnx"


def main():
    # Dynamic quantization — kalibrasyon verisi gerektirmez, hızlı ve etkili
    quantize_dynamic(
        model_input=ONNX_FP32,
        model_output=ONNX_INT8,
        weight_type=QuantType.QInt8,
    )

    # INT8 model doğrulama
    sess = ort.InferenceSession(ONNX_INT8, providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: dummy_input})[0]
    print(f"✓ INT8 ONNX Runtime çalıştırma başarılı, çıktı shape: {out.shape}")

    fp32_mb = os.path.getsize(ONNX_FP32) / 1e6
    int8_mb = os.path.getsize(ONNX_INT8) / 1e6
    print(f"\nFP32 boyutu : {fp32_mb:.1f} MB")
    print(f"INT8 boyutu : {int8_mb:.1f} MB")
    print(f"Küçülme     : {fp32_mb/int8_mb:.1f}×")
    print(f"✓ Kaydedildi: {ONNX_INT8}")


if __name__ == "__main__":
    main()