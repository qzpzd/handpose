"""
推理前向性能基准（不依赖摄像头，不依赖 cv2）

说明：
1) 仅测“模型前向推理”耗时，不包含图像预处理/后处理（因为当前环境下 cv2 无法导入，且 test_camera_* 脚本依赖 cv2）。
2) ONNX 使用 onnxruntime，TFLite 使用 tf.lite.Interpreter。
3) TensorRT 若 tensorrt/pycuda 环境不可用则跳过。
"""

from __future__ import annotations

import os
import time
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PALM_ONNX = os.path.join(BASE_DIR, "palm_detection_lite_nchw.onnx")
HAND_ONNX = os.path.join(BASE_DIR, "hand_landmark_lite_nchw.onnx")

PALM_TFLITE = os.path.join(BASE_DIR, "palm_detection_lite.tflite")
HAND_TFLITE = os.path.join(BASE_DIR, "hand_landmark_lite.tflite")

PALM_TRT = os.path.join(BASE_DIR, "palm_detection_lite_nchw.engine")
HAND_TRT = os.path.join(BASE_DIR, "hand_landmark_lite_nchw.engine")


WARMUP = int(os.environ.get("BENCH_WARMUP", "10"))
ITER = int(os.environ.get("BENCH_ITER", "100"))


def _bench_time(fn, warmup: int, iters: int) -> float:
    # warmup
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def _make_onnx_dummy(shape) -> np.ndarray:
    # onnxruntime 里 shape 可能包含 None / -1
    fixed = []
    for d in shape:
        if d is None or d == -1:
            fixed.append(1)
        else:
            fixed.append(int(d))
    # 统一 float32，符合大多数 onnx 模型输入
    return np.random.rand(*fixed).astype(np.float32)


def bench_onnx_combined() -> dict:
    import onnxruntime as ort

    providers_req = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    palm_sess = ort.InferenceSession(PALM_ONNX, providers=providers_req)
    hand_sess = ort.InferenceSession(HAND_ONNX, providers=providers_req)

    palm_in = palm_sess.get_inputs()[0].name
    palm_shape = palm_sess.get_inputs()[0].shape
    palm_dummy = _make_onnx_dummy(palm_shape)

    hand_in = hand_sess.get_inputs()[0].name
    hand_shape = hand_sess.get_inputs()[0].shape
    hand_dummy = _make_onnx_dummy(hand_shape)

    # 实际使用的 providers（非常关键：当前 onnxruntime 可能不带 CUDA）
    providers_actual = {
        "palm": palm_sess.get_providers(),
        "hand": hand_sess.get_providers(),
    }

    def palm_once():
        palm_sess.run(None, {palm_in: palm_dummy})

    def hand_once():
        hand_sess.run(None, {hand_in: hand_dummy})

    def combined_once():
        palm_sess.run(None, {palm_in: palm_dummy})
        hand_sess.run(None, {hand_in: hand_dummy})

    palm_ms = _bench_time(palm_once, WARMUP, ITER)
    hand_ms = _bench_time(hand_once, WARMUP, ITER)
    total_ms = _bench_time(combined_once, WARMUP, ITER)

    return {
        "backend": "ONNX Runtime (front-end only)",
        "providers_actual": providers_actual,
        "palm_ms": palm_ms,
        "hand_ms": hand_ms,
        "total_ms": total_ms,
        "fps": 1000.0 / total_ms if total_ms > 0 else None,
    }


def _make_tflite_dummy(input_detail: dict) -> np.ndarray:
    shape = input_detail["shape"]
    dtype = input_detail["dtype"]

    fixed = []
    for d in shape:
        if d is None or d == -1:
            fixed.append(1)
        else:
            fixed.append(int(d))

    if np.issubdtype(dtype, np.floating):
        return np.random.rand(*fixed).astype(dtype)
    else:
        # quantized / integer 输入
        info = np.iinfo(dtype)
        return np.random.randint(info.min, info.max + 1, size=fixed, dtype=dtype)


def bench_tflite_combined() -> dict:
    import tensorflow as tf

    palm_int = tf.lite.Interpreter(model_path=PALM_TFLITE)
    hand_int = tf.lite.Interpreter(model_path=HAND_TFLITE)
    palm_int.allocate_tensors()
    hand_int.allocate_tensors()

    palm_in = palm_int.get_input_details()[0]
    hand_in = hand_int.get_input_details()[0]

    palm_dummy = _make_tflite_dummy(palm_in)
    hand_dummy = _make_tflite_dummy(hand_in)

    def palm_once():
        palm_int.set_tensor(palm_in["index"], palm_dummy)
        palm_int.invoke()

    def hand_once():
        hand_int.set_tensor(hand_in["index"], hand_dummy)
        hand_int.invoke()

    def combined_once():
        palm_int.set_tensor(palm_in["index"], palm_dummy)
        palm_int.invoke()
        hand_int.set_tensor(hand_in["index"], hand_dummy)
        hand_int.invoke()

    palm_ms = _bench_time(palm_once, WARMUP, ITER)
    hand_ms = _bench_time(hand_once, WARMUP, ITER)
    total_ms = _bench_time(combined_once, WARMUP, ITER)

    gpus = tf.config.list_physical_devices("GPU")

    return {
        "backend": "TensorFlow Lite (front-end only)",
        "tf_gpus": [str(x) for x in gpus],
        "palm_ms": palm_ms,
        "hand_ms": hand_ms,
        "total_ms": total_ms,
        "fps": 1000.0 / total_ms if total_ms > 0 else None,
    }


def bench_tensorrt_combined() -> dict:
    """
    仅在环境具备 tensorrt + pycuda 时测试。
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except Exception as e:
        return {
            "backend": "TensorRT (front-end only)",
            "error": f"环境不可用：{e}",
        }

    # tensorrt 代码较复杂（需要绑定内存、设置 binding 等）。
    # 为了避免引入 cv2 和保持脚本可读性，这里只给出“可用性检测”与跳过。
    # 如果你确认环境已安装 tensorrt/pycuda，我可以继续补齐 TRT 的完整 micro-benchmark。
    return {
        "backend": "TensorRT (front-end only)",
        "status": "环境已安装 tensorrt/pycuda，但当前脚本尚未实现 TRT micro-benchmark 细节。请告知我继续补齐。"
    }


def main():
    print(f"Warmup={WARMUP}, Iter={ITER}")

    results = []
    for fn in [bench_onnx_combined, bench_tflite_combined, bench_tensorrt_combined]:
        r = fn()
        results.append(r)
        print("\n" + "=" * 60)
        print(r.get("backend"))
        for k, v in r.items():
            if k == "backend":
                continue
            print(f"{k}: {v}")

    return results


if __name__ == "__main__":
    main()

