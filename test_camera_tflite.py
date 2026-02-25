"""
TFLite 格式模型摄像头测试
==========================

使用 TFLite 格式的模型进行实时推理
"""

import cv2
import numpy as np
import time
import os
import platform
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mp_palmdet_tflite import MPPalmDetTFLite
from mp_handpose_tflite import MPHandPoseTFLite

try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except Exception as e:
    TFLITE_AVAILABLE = False
    print(f"⚠️ TensorFlow 导入失败: {e}")
    print("TFLite 模型测试不可用")

# 检测操作系统
IS_WINDOWS = platform.system() == "Windows"

# 摄像头配置
USB_DEVICE_ID = 1
CAMERA_FRAMERATE = 30
DISPLAY_SIZE = (640, 640)

# 模型配置
PalmDet_MODEL = "palm_detection_lite"
HandLandmark_MODEL = "hand_landmark_lite"

def draw_landmarks(image, landmarks):
    """绘制手部关键点"""
    for i in range(0, 63, 3):
        x, y, z = landmarks[i], landmarks[i+1], landmarks[i+2]
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    # 绘制手部连接线
    connections = [[0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
                   [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
                   [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
                   [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
                   [0, 17], [17, 18], [18, 19], [19, 20]]  # 小指
    
    for conn in connections:
        start_idx = conn[0] * 3
        end_idx = conn[1] * 3
        start_point = (int(landmarks[start_idx]), int(landmarks[start_idx+1]))
        end_point = (int(landmarks[end_idx]), int(landmarks[end_idx+1]))
        cv2.line(image, start_point, end_point, (255, 0, 0), 2)

def process_detection_results(frame, palms, hands):
    """处理检测结果并绘制"""
    # 绘制手掌边界框
    for palm in palms:
        bbox = palm['bbox'].astype(int)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # 绘制手部结果
    for hand_results in hands:
        if hand_results is not None:
            # 绘制手部边界框（使用返回的 bbox）
            hand_x1, hand_y1, hand_x2, hand_y2 = hand_results[:4].astype(int)
            cv2.rectangle(frame, (hand_x1, hand_y1), (hand_x2, hand_y2), (255, 0, 255), 2)
            
            # 绘制手部关键点（screen landmarks）
            landmarks = hand_results[4:67]
            draw_landmarks(frame, landmarks)
            
            # 显示左右手信息和得分
            handedness = hand_results[130]
            conf = hand_results[131]
            hand_label = "Right" if handedness > 0.5 else "Left"
            cv2.putText(frame, f"Hand: {hand_label}", (hand_x1, hand_y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Score: {conf:.2f}", (hand_x1, hand_y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    if not TFLITE_AVAILABLE:
        print("❌ TensorFlow 未安装，无法运行 TFLite 模型测试")
        print("请安装 TensorFlow: pip install tensorflow")
        return
    
    print("=" * 80)
    print("TFLite 格式模型摄像头测试")
    print("=" * 80)
    
    base_dir = r"D:\ai_projects\llm_sop\knowledge-assistant\palm_hand"
    
    # 加载模型 - 使用 TFLite 模型类
    print(f"\n加载模型...")
    try:
        palm_detector = MPPalmDetTFLite(os.path.join(base_dir, f"{PalmDet_MODEL}.tflite"), use_gpu=True, scoreThreshold=0.3)
        hand_detector = MPHandPoseTFLite(os.path.join(base_dir, f"{HandLandmark_MODEL}.tflite"), use_gpu=True, confThreshold=0.3)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 初始化摄像头
    if IS_WINDOWS:
        print(f"\n打开摄像头 (设备ID: {USB_DEVICE_ID})...")
        cap = cv2.VideoCapture(USB_DEVICE_ID, cv2.CAP_DSHOW)
    else:
        print(f"\n打开摄像头 (设备ID: {USB_DEVICE_ID})...")
        cap = cv2.VideoCapture(USB_DEVICE_ID)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FRAMERATE)
    
    if not cap.isOpened():
        raise Exception(f"无法打开摄像头 (设备ID: {USB_DEVICE_ID})")
    
    print(f"✓ 摄像头打开成功")
    print(f"✓ 开始实时推理，按 'q' 退出...")
    print(f"✓ 模型格式: ONNX (使用 TFLite 模型文件)")
    
    fps = 0
    frame_count = 0
    total_time = 0
    start_time = time.time()
    
    while True:
        loop_start = time.time()
        
        # 捕获帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头帧，跳过...")
            continue
        
        # 记录推理时间
        inference_start = time.time()
        
        # 手掌检测
        palms = palm_detector.infer(frame)
        
        # 手部关键点检测
        hands = []
        if palms is not None and len(palms) > 0:
            for palm in palms:
                handpose = hand_detector.infer(frame, palm)
                if handpose is not None:
                    hands.append(handpose)
        
        inference_time = time.time() - inference_start
        total_time += inference_time
        frame_count += 1
        
        # 处理检测结果并绘制
        process_detection_results(frame, palms, hands)
        
        # 显示信息
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_inference_time = total_time / frame_count if frame_count > 0 else 0
        
        cv2.putText(frame, f"Model: TFLite", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Detections: {len(palms) if palms is not None else 0}, Hands: {len(hands)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Inference: {avg_inference_time*1000:.1f}ms", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Palm & Hand Detection (ONNX)", frame)
        
        # 计算FPS
        loop_time = time.time() - loop_start
        if loop_time > 0:
            instant_fps = 1 / loop_time
            fps = 0.9 * fps + 0.1 * instant_fps
        
        frame_count += 1
        
        # 退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ 测试结束，共处理 {frame_count} 帧")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 程序运行异常：{str(e)}")
        import traceback
        traceback.print_exc()
