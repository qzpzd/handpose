"""
三后端端到端推理性能基准（无需摄像头）
在 test-gpu 环境运行:
  D:\\files\\conda_env\\test-gpu\\python.exe benchmark_backends.py
"""
import os
import sys
import time
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

WARMUP = 10
ITERATIONS = 100
FRAME_SIZE = (640, 640, 3)


def make_test_frame():
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, size=FRAME_SIZE, dtype=np.uint8)
    # 肤色区域，提高检出概率
    frame[200:450, 200:450] = [180, 140, 120]
    return frame


def print_result(r):
    print(f"\n{'=' * 60}")
    print(f"后端: {r['backend']}")
    print(f"执行设备/提供者: {r.get('device')}")
    print(f"GPU 实际启用: {'是' if r['gpu_active'] else '否'}")
    if r.get("note"):
        print(f"说明: {r['note']}")
    print(f"手掌检测: {r['palm_ms']:.2f} ms")
    print(f"手部关键点: {r['hand_ms']:.2f} ms")
    print(f"端到端推理: {r['total_ms']:.2f} ms")
    print(f"等效 FPS: {r['fps']:.1f}")
    print(f"平均手掌数/帧: {r['avg_palms']:.2f}")
    print(f"平均手数/帧: {r['avg_hands']:.2f}")


def bench_onnx(frame):
    from mp_palmdet import MPPalmDet
    from mp_handpose import MPHandPose

    palm_path = os.path.join(BASE_DIR, "palm_detection_lite_nchw.onnx")
    hand_path = os.path.join(BASE_DIR, "hand_landmark_lite_nchw.onnx")

    palm_det = MPPalmDet(palm_path, use_gpu=True, scoreThreshold=0.3)
    hand_pose = MPHandPose(hand_path, use_gpu=True, confThreshold=0.3)

    providers = palm_det.session.get_providers()
    gpu_active = any(p in providers for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider"))

    for _ in range(WARMUP):
        palms = palm_det.infer(frame)
        if palms is not None and len(palms) > 0:
            hand_pose.infer(frame, palms[0])

    palm_times, hand_times, total_times = [], [], []
    palm_count = hand_count = 0

    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        t_palm = time.perf_counter()
        palms = palm_det.infer(frame)
        palm_times.append((time.perf_counter() - t_palm) * 1000)
        n = 0 if palms is None else len(palms)
        palm_count += n

        hand_ms = 0.0
        if n > 0:
            t_hand = time.perf_counter()
            for palm in palms:
                if hand_pose.infer(frame, palm) is not None:
                    hand_count += 1
            hand_ms = (time.perf_counter() - t_hand) * 1000
        hand_times.append(hand_ms)
        total_times.append((time.perf_counter() - t0) * 1000)

    return {
        "backend": "ONNX Runtime",
        "device": providers,
        "gpu_active": gpu_active,
        "palm_ms": float(np.mean(palm_times)),
        "hand_ms": float(np.mean(hand_times)),
        "total_ms": float(np.mean(total_times)),
        "fps": 1000.0 / float(np.mean(total_times)),
        "avg_palms": palm_count / ITERATIONS,
        "avg_hands": hand_count / ITERATIONS,
    }


def bench_tflite(frame):
    from mp_palmdet_tflite import MPPalmDetTFLite
    from mp_handpose_tflite import MPHandPoseTFLite

    palm_path = os.path.join(BASE_DIR, "palm_detection_lite.tflite")
    hand_path = os.path.join(BASE_DIR, "hand_landmark_lite.tflite")

    palm_det = MPPalmDetTFLite(palm_path, use_gpu=True, scoreThreshold=0.3)
    hand_pose = MPHandPoseTFLite(hand_path, use_gpu=True, confThreshold=0.3)

    # 代码里 use_gpu 只是保存参数，未加载任何 GPU delegate
    note = "use_gpu=True 但未实现 GPU delegate，实际 CPU/XNNPACK"

    for _ in range(WARMUP):
        palms = palm_det.infer(frame)
        if palms:
            hand_pose.infer(frame, palms[0])

    palm_times, hand_times, total_times = [], [], []
    palm_count = hand_count = 0

    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        t_palm = time.perf_counter()
        palms = palm_det.infer(frame)
        palm_times.append((time.perf_counter() - t_palm) * 1000)
        n = 0 if not palms else len(palms)
        palm_count += n

        hand_ms = 0.0
        if n > 0:
            t_hand = time.perf_counter()
            for palm in palms:
                if hand_pose.infer(frame, palm) is not None:
                    hand_count += 1
            hand_ms = (time.perf_counter() - t_hand) * 1000
        hand_times.append(hand_ms)
        total_times.append((time.perf_counter() - t0) * 1000)

    return {
        "backend": "TensorFlow Lite",
        "device": "CPU (TFLite Interpreter / XNNPACK)",
        "gpu_active": False,
        "note": note,
        "palm_ms": float(np.mean(palm_times)),
        "hand_ms": float(np.mean(hand_times)),
        "total_ms": float(np.mean(total_times)),
        "fps": 1000.0 / float(np.mean(total_times)),
        "avg_palms": palm_count / ITERATIONS,
        "avg_hands": hand_count / ITERATIONS,
    }


def bench_tensorrt(frame):
    from mp_palmdet_trt import MPPalmDetTRT
    from mp_handpose_trt import MPHandPoseTRT

    palm_path = os.path.join(BASE_DIR, "palm_detection_lite_nchw.engine")
    hand_path = os.path.join(BASE_DIR, "hand_landmark_lite_nchw.engine")

    palm_det = MPPalmDetTRT(palm_path, use_gpu=True, scoreThreshold=0.3)
    hand_pose = MPHandPoseTRT(hand_path, use_gpu=True, confThreshold=0.1)

    for _ in range(WARMUP):
        palms = palm_det.infer(frame)
        if palms is not None and len(palms) > 0:
            hand_pose.infer(frame, palms[0])

    palm_times, hand_times, total_times = [], [], []
    palm_count = hand_count = 0

    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        t_palm = time.perf_counter()
        palms = palm_det.infer(frame)
        palm_times.append((time.perf_counter() - t_palm) * 1000)
        n = 0 if palms is None else len(palms)
        palm_count += n

        hand_ms = 0.0
        if n > 0:
            t_hand = time.perf_counter()
            for palm in palms:
                if hand_pose.infer(frame, palm) is not None:
                    hand_count += 1
            hand_ms = (time.perf_counter() - t_hand) * 1000
        hand_times.append(hand_ms)
        total_times.append((time.perf_counter() - t0) * 1000)

    return {
        "backend": "TensorRT",
        "device": "CUDA (TensorRT Engine + PyCUDA)",
        "gpu_active": True,
        "note": "原生 GPU 推理，use_gpu 参数本身未被分支使用",
        "palm_ms": float(np.mean(palm_times)),
        "hand_ms": float(np.mean(hand_times)),
        "total_ms": float(np.mean(total_times)),
        "fps": 1000.0 / float(np.mean(total_times)),
        "avg_palms": palm_count / ITERATIONS,
        "avg_hands": hand_count / ITERATIONS,
    }


def main():
    print("Hand Pose 三后端性能基准测试")
    print(f"Python: {sys.executable}")
    print(f"测试帧: {FRAME_SIZE[1]}x{FRAME_SIZE[0]}, 预热 {WARMUP}, 测试 {ITERATIONS}")
    frame = make_test_frame()

    results = []
    for name, fn in [
        ("ONNX", bench_onnx),
        ("TFLite", bench_tflite),
        ("TensorRT", bench_tensorrt),
    ]:
        print(f"\n>>> 正在测试 {name}...")
        try:
            r = fn(frame)
            results.append(r)
            print_result(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"backend": name, "error": str(e)})
            print(f"  [失败] {name}: {e}")

    print(f"\n{'=' * 60}")
    print("汇总对比:")
    print(f"{'后端':<20} {'GPU':<8} {'推理(ms)':<12} {'FPS':<8}")
    print("-" * 52)
    for r in results:
        if "error" in r:
            print(f"{r['backend']:<20} {'N/A':<8} {'失败':<12} {'N/A':<8}")
        else:
            gpu = "是" if r["gpu_active"] else "否"
            print(f"{r['backend']:<20} {gpu:<8} {r['total_ms']:.1f}{'':<7} {r['fps']:.1f}")


if __name__ == "__main__":
    main()
