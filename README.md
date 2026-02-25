# Hand Pose Estimation (MediaPipe)

基于 MediaPipe 的实时手势姿态估计项目，支持多种推理后端（ONNX、TensorFlow Lite、TensorRT）。

## 项目结构

### 核心推理模块

| 文件 | 说明 |
|------|------|
| [mp_palmdet.py](file:///d:\qzp\handpose\mp_palmdet.py) | 手掌检测 ONNX 推理模块 |
| [mp_handpose.py](file:///d:\qzp\handpose\mp_handpose.py) | 手部关键点估计 ONNX 推理模块 |
| [mp_palmdet_tflite.py](file:///d:\qzp\handpose\mp_palmdet_tflite.py) | 手掌检测 TensorFlow Lite 推理模块 |
| [mp_handpose_tflite.py](file:///d:\qzp\handpose\mp_handpose_tflite.py) | 手部关键点估计 TensorFlow Lite 推理模块 |
| [mp_palmdet_trt.py](file:///d:\qzp\handpose\mp_palmdet_trt.py) | 手掌检测 TensorRT 推理模块 |
| [mp_handpose_trt.py](file:///d:\qzp\handpose\mp_handpose_trt.py) | 手部关键点估计 TensorRT 推理模块 |

### 模型转换脚本

| 文件 | 说明 |
|------|------|
| [tflite_to_onnx.py](file:///d:\qzp\handpose\tflite_to_onnx.py) | 将 TFLite 模型转换为 ONNX 格式 |
| [convert_onnx_nchw.py](file:///d:\qzp\handpose\convert_onnx_nchw.py) | 将 ONNX 模型从 NHWC 转换为 NCHW 格式 |
| [convert_to_trt.py](file:///d:\qzp\handpose\convert_to_trt.py) | 将 ONNX 模型转换为 TensorRT Engine |

### 测试与演示脚本

| 文件 | 说明 |
|------|------|
| [test_camera_onnx.py](file:///d:\qzp\handpose\test_camera_onnx.py) | 使用 ONNX 模型测试摄像头实时推理 |
| [test_camera_tflite.py](file:///d:\qzp\handpose\test_camera_tflite.py) | 使用 TFLite 模型测试摄像头实时推理 |
| [test_camera_tensorrt.py](file:///d:\qzp\handpose\test_camera_tensorrt.py) | 使用 TensorRT 模型测试摄像头实时推理 |
| [onnx_demo_final.py](file:///d:\qzp\handpose\onnx_demo_final.py) | ONNX 模型最终演示 |

### 模型文件

| 文件 | 说明 |
|------|------|
| `palm_detection_mediapipe_2023feb.onnx` | MediaPipe 手掌检测原始 ONNX 模型 |
| `palm_detection_mediapipe_2023feb_int8.onnx` | INT8 量化手掌检测模型 |
| `palm_detection_mediapipe_2023feb_int8bq.onnx` | INT8 量化手掌检测模型（批量化） |
| `palm_detection_lite_nchw.onnx` | NCHW 格式手掌检测模型 |
| `palm_detection_lite_nchw.engine` | TensorRT 引擎手掌检测模型 |
| `palm_detection_lite.tflite` | TFLite 手掌检测模型 |
| `hand_landmark_lite_nchw.onnx` | NCHW 格式手部关键点模型 |
| `hand_landmark_lite_nchw.engine` | TensorRT 引擎手部关键点模型 |
| `hand_landmark_lite.tflite` | TFLite 手部关键点模型 |

## 依赖安装

```bash
pip install opencv-python numpy onnxruntime-gpu tensorrt
pip install tensorflow tf2onnx onnx onnxruntime
pip install pycuda
```

## 使用方法

### 1. ONNX 模型实时测试

```bash
python test_camera_onnx.py
```

### 2. TensorFlow Lite 模型实时测试

```bash
python test_camera_tflite.py
```

### 3. TensorRT 模型实时测试

```bash
python test_camera_tensorrt.py
```

### 4. 模型转换

#### TFLite 转 ONNX

```bash
python tflite_to_onnx.py
```

#### ONNX NHWC 转 NCHW

```bash
python convert_onnx_nchw.py
```

#### ONNX 转 TensorRT

```bash
python convert_to_trt.py
```

## 各推理后端对比与总结

| 特性 | ONNX Runtime | TensorFlow Lite | TensorRT |
|------|-------------|-----------------|----------|
| **精度** | FP32/FP16/INT8 | FP32/FP16/INT8 | FP32/FP16/INT8 |
| **速度** | 中等 | 较快 | 最快 |
| **兼容性** | 广泛 | Android/iOS/嵌入式 | NVIDIA GPU |
| **易用性** | 简单 | 简单 | 较复杂 |
| **GPU 加速** | CUDA | GPU Delegate | 原生支持 |
| **模型大小** | 较大 | 较小 | 较大 |

### 推荐使用场景

| 场景 | 推荐后端 |
|------|---------|
| **快速原型开发** | ONNX Runtime |
| **移动端/嵌入式** | TensorFlow Lite |
| **高性能推理 (NVIDIA GPU)** | TensorRT |
| **服务器端部署** | ONNX Runtime 或 TensorRT |

### 性能优化建议

1. **使用 TensorRT**: 在 NVIDIA GPU 上使用 TensorRT 可获得最佳性能
2. **模型量化**: INT8 量化可显著减少模型体积和推理时间
3. **批处理**: 多个输入一起处理可提高吞吐量
4. **NCHW 格式**: TensorRT 建议使用 NCHW 格式以获得更好性能

## 关键参数说明

### 手掌检测参数 (MPPalmDet)

- `nmsThreshold`: 非极大值抑制阈值，默认 0.3
- `scoreThreshold`: 置信度阈值，默认 0.5
- `topK`: 最大检测数量，默认 5000

### 手部关键点参数 (MPHandPose)

- `confThreshold`: 关键点置信度阈值，默认 0.8
- `input_size`: 输入图像尺寸，默认为 224x224

## 注意事项

1. 使用 TensorRT 需要 NVIDIA 显卡和 CUDA 环境
2. TFLite 模型主要面向移动端和嵌入式设备
3. ONNX 模型可在多种平台上运行，兼容性最好
4. INT8 量化模型精度可能略有下降
