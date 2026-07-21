# 手势推理性能与 TensorRT 模块说明

本文档汇总三后端实测结果，并说明 `mp_palmdet_trt.py`、`mp_handpose_trt.py` 的实现逻辑。

---

## 1. 测试环境

| 项目 | 值 |
|------|----|
| 环境 | `D:\files\conda_env\test-gpu` |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB) |
| ONNX Runtime | 1.18.1（含 `CUDAExecutionProvider`） |
| TensorRT | 8.6.1 + PyCUDA |
| TensorFlow | 2.20.0（无 GPU 设备列表） |
| 测试素材 | `hand_test.mp4`（检出 1 只手） |
| 统计方式 | 预热 10 次，计时 80 次，含预处理 + 前向 + 后处理 |

推荐运行命令：

```powershell
D:\files\conda_env\test-gpu\python.exe test_camera_onnx.py
D:\files\conda_env\test-gpu\python.exe test_camera_tflite.py
D:\files\conda_env\test-gpu\python.exe test_camera_tensorrt.py
```

---

## 2. 模型输入尺寸（ONNX / TensorRT 一致）

| 模型 | 文件 | 输入名 | 输入形状 | 精度 |
|------|------|--------|----------|------|
| 手掌检测 | `palm_detection_lite_nchw.onnx` / `.engine` | `input_1` | `[1, 3, 192, 192]` | float32 |
| 手部关键点 | `hand_landmark_lite_nchw.onnx` / `.engine` | `input_1` | `[1, 3, 224, 224]` | float32 |

说明：均为 **NCHW**，batch=1，RGB，数值范围约 `[0, 1]`（代码中 `/255`）。

### 输出张量

**手掌检测**

| 名称 | 形状 | 含义 |
|------|------|------|
| `Identity` | `[1, 2016, 18]` | box(4) + 7 个关键点(14) |
| `Identity_1` | `[1, 2016, 1]` | 原始 score（需 sigmoid） |

**手部关键点**

| 名称 | 形状 | 含义 |
|------|------|------|
| `Identity` | `[1, 63]` | 屏幕坐标关键点（21×3） |
| `Identity_1` | `[1, 1]` | 置信度 |
| `Identity_2` | `[1, 1]` | 左右手 |
| `Identity_3` | `[1, 63]` | 世界坐标关键点（21×3） |

注意：TensorRT engine 的 binding **顺序**可能与 ONNX 不同，代码按 **名称** 取输出，避免顺序错乱。

---

## 3. 三后端 GPU 使用情况

| 测试脚本 | 代码意图 | 实际是否用 GPU |
|----------|----------|----------------|
| `test_camera_onnx.py` | `use_gpu=True` → CUDA EP | **是**（`CUDAExecutionProvider`） |
| `test_camera_tflite.py` | 传入 `use_gpu=True` | **否**（未实现 GPU delegate，走 CPU/XNNPACK） |
| `test_camera_tensorrt.py` | TensorRT + PyCUDA | **是**（原生 CUDA；`use_gpu` 参数未分支） |

要点：

- 默认 `miniconda3` 环境常只有 CPU 版 `onnxruntime`，GPU 推理请用 `test-gpu`。
- TFLite 模块里 `use_gpu` 仅保存，未调用 GPU delegate。

---

## 4. 性能对比结果

条件：视频中检出 1 只手，端到端（检测 + 关键点）。

| 后端 | GPU | 手掌检测 | 关键点 | 端到端 | 等效 FPS |
|------|-----|----------|------|--------|----------|
| ONNX Runtime | 是 | 5.75 ms | 5.76 ms | **11.51 ms** | **86.9** |
| TensorFlow Lite | 否 | 14.88 ms | 7.91 ms | **22.82 ms** | **43.8** |
| TensorRT | 是 | 2.41 ms | 0.91 ms | **3.33 ms** | **~300** |

相对速度（端到端）：**TensorRT ≈ 3.5× ONNX ≈ 6.9× TFLite**。

### 结论

1. **TensorRT** 在本机 NVIDIA GPU 上最快，适合高性能部署。
2. **ONNX + CUDA** 居中，易用，实时约 80+ FPS。
3. **TFLite** 当前未用 GPU，最慢；适合移动端/嵌入式，桌面 GPU 场景优先选 ONNX/TRT。

---

## 5. 整体推理流水线

```text
摄像头/视频帧
    │
    ▼
┌─────────────────────┐
│  MPPalmDetTRT       │  输入 192×192 NCHW
│  手掌检测 + NMS     │  输出: bbox + 7 掌部关键点 + score
└─────────┬───────────┘
          │ 每个 palm
          ▼
┌─────────────────────┐
│  MPHandPoseTRT      │  按 palm 裁剪/旋转 → 224×224
│  21 点关键点         │  输出: 手框 + 63 屏坐标 + 63 世界坐标 + 左右手 + conf
└─────────────────────┘
```

`test_camera_tensorrt.py` 即按上述流程循环调用两个模块。

---

## 6. `mp_palmdet_trt.py` 代码说明

文件职责：加载手掌检测 TensorRT Engine，完成预处理、GPU 推理、NMS 后处理。

### 6.1 类与初始化 `MPPalmDetTRT.__init__`

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `modelPath` | — | `.engine` 路径 |
| `nmsThreshold` | 0.3 | NMS IoU 阈值 |
| `scoreThreshold` | 0.5 | 置信度阈值 |
| `topK` | 5000 | 保留上限（接口保留） |
| `use_gpu` | True | 仅保存；TRT 本身始终走 CUDA |

初始化步骤：

1. **反序列化 Engine**：`trt.Runtime.deserialize_cuda_engine`
2. **创建执行上下文**：`create_execution_context`
3. **记录输出 binding 名与 shape**（跳过 index 0 输入）
4. **分配 Host/Device 内存**：page-locked host + `cuda.mem_alloc` device，填入 `bindings`
5. **加载 anchors**：`_load_anchors()`（与 MediaPipe 手掌检测 anchor 表一致，文件后半为大表）

内存布局约定：`inputs` / `outputs` 均按 `[host, device, host, device, ...]` 成对存放。

### 6.2 预处理 `_preprocess`

1. 按短边比例缩放到适应 `192×192`，保持宽高比。
2. 四周 pad 到 `192×192`，记录 `pad_bias`（再除以 ratio 映射回原图坐标系）。
3. BGR→RGB，`/255` 归一化。
4. `HWC → NCHW`，得到 `[1, 3, 192, 192]`。

### 6.3 推理 `infer`

```text
预处理 → HtoD(async) → execute_async_v2 → DtoH(async) → synchronize
→ 按输出名组装 output_dict → _postprocess
```

关键点：异步拷贝与执行共用同一 `cuda.Stream`，最后 `synchronize` 保证结果可读。

### 6.4 后处理 `_postprocess`

1. 按名取 `Identity`（18 维回归）与 `Identity_1`（score）。
2. 拆分 `box_delta` / `landmark_delta`。
3. 结合 anchors 解码框与 7 个掌部关键点，乘以 `scale = max(w, h)`。
4. score 做 sigmoid，按 `scoreThreshold` 过滤。
5. `cv.dnn.NMSBoxes` 去重。
6. 减去 `pad_bias`，还原到原图像素坐标。

**单条结果格式**（长度 19）：

```text
[x1, y1, x2, y2,  lm0x, lm0y, ..., lm6x, lm6y,  score]
 │── bbox 4 ──│  │────── 7×2 关键点 14 ──────│  │ 1 │
```

返回类型为 **list of ndarray**（与部分 ONNX 路径略有差异，下游需兼容）。

### 6.5 主要方法一览

| 方法 | 作用 |
|------|------|
| `__init__` | 加载 engine、分配显存、加载 anchors |
| `_preprocess` | 缩放/pad/归一化/NCHW |
| `infer` | 端到端推理入口 |
| `_postprocess` | 解码框 + NMS + 坐标还原 |
| `_load_anchors` | 返回固定 anchor 数组 |

---

## 7. `mp_handpose_trt.py` 代码说明

文件职责：在已检测到的手掌框上，用 TensorRT 估计 21 点手部关键点。

### 7.1 类与初始化 `MPHandPoseTRT.__init__`

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `modelPath` | — | `hand_landmark_lite_nchw.engine` |
| `confThreshold` | 0.8 | 低于此置信度返回 `None` |
| `use_gpu` | True | 仅保存；实际始终 CUDA |

几何常量（与 MediaPipe 对齐）：

- 旋转前 enlarge：`PALM_BOX_PRE_ENLARGE_FACTOR = 4`
- 旋转后 enlarge：`PALM_BOX_ENLARGE_FACTOR = 3`
- 手框 shift / enlarge：`HAND_BOX_SHIFT_VECTOR`、`HAND_BOX_ENLARGE_FACTOR`
- 输入尺寸：`224×224`

Engine 加载与显存分配逻辑与 `MPPalmDetTRT` 相同。

### 7.2 裁剪与填充 `_cropAndPadFromPalm`

对 palm bbox：

1. 按模式选择 shift / enlarge（旋转前 vs 旋转后）。
2. clip 到图像范围后 crop。
3. pad 成正方形（旋转前用对角线长度，旋转后用 max 边长）。
4. 返回 crop 图、bbox、`bias`（用于坐标回映射）。

空 crop 时返回 `None`，上层直接放弃该手。

### 7.3 预处理 `_preprocess`

输入：`image` + `palm`（来自手掌检测的 19 维向量）。

流程：

1. 用 palm bbox **旋转前**裁剪放大。
2. BGR→RGB。
3. 用掌根与中指根关键点算旋转角，将手摆正（`warpAffine`）。
4. 在旋转图上再裁一次，resize 到 `224×224`，`/255`，转 NCHW。
5. 返回：`input_blob`、`rotated_palm_bbox`、`angle`、`rotation_matrix`、`pad_bias`。

失败时返回一组 `None`，`infer` 直接返回 `None`。

### 7.4 推理 `infer`

与手掌模块相同的 TRT 异步路径，再 `_postprocess`。

### 7.5 后处理 `_postprocess`

1. 按名取四个输出：`Identity` / `Identity_1` / `Identity_2` / `Identity_3`。
2. `conf < confThreshold` → `None`。
3. 将局部关键点缩放、逆旋转、加回 `original_center + pad_bias`，映射到原图。
4. 由关键点生成手部 bbox，再 shift/enlarge。

**返回向量结构**（长度 132）：

```text
[0:4]     手部 bbox [x1,y1,x2,y2]
[4:67]    屏幕关键点 21×3
[67:130]  世界关键点 21×3
[130]     handedness（>0.5 偏右手）
[131]     confidence
```

### 7.6 主要方法一览

| 方法 | 作用 |
|------|------|
| `__init__` | 加载 landmark engine、分配显存 |
| `_cropAndPadFromPalm` | 按掌框裁剪并 pad 方形 |
| `_preprocess` | 旋转对齐 + 生成 224 NCHW 输入 |
| `infer` | GPU 推理入口 |
| `_postprocess` | 坐标还原与手框生成 |

---

## 8. 两模块协作关系

```text
MPPalmDetTRT.infer(frame)
        │
        │  list[ ndarray(19,) ]
        ▼
for palm in palms:
    MPHandPoseTRT.infer(frame, palm)
        │
        │  ndarray(132,) 或 None
        ▼
绘制 bbox / 21 点骨架 / Left-Right / Score
```

依赖：

- `tensorrt`、`pycuda`（`pycuda.autoinit` 初始化 CUDA 上下文）
- `opencv-python`、`numpy`
- 与 GPU/驱动匹配的 `.engine`（换卡通常需重新 `convert_to_trt.py`）

---

## 9. 使用注意

1. **Engine 与 GPU 绑定**：TensorRT engine 与构建时的 GPU 架构相关，换机/换卡请重新转换。
2. **输出顺序**：务必按 tensor **名称**解析，不要写死 index。
3. **阈值**：摄像头脚本里手掌常用 `scoreThreshold=0.3`，关键点 TRT 可用较低 `confThreshold`（如 0.1）以提高召回。
4. **返回类型**：`MPPalmDetTRT.infer` 返回 list；ONNX 版可能返回 ndarray，接入时注意统一。
5. **`use_gpu`**：TRT 两模块中该参数无实际分支，仅接口兼容。

---

## 10. 相关文件

| 文件 | 说明 |
|------|------|
| `mp_palmdet_trt.py` | 手掌 TensorRT 推理 |
| `mp_handpose_trt.py` | 关键点 TensorRT 推理 |
| `convert_to_trt.py` | ONNX → Engine |
| `test_camera_tensorrt.py` | 摄像头联调 |
| `benchmark_backends.py` | 三后端基准脚本 |
| `README.md` | 项目总览 |
