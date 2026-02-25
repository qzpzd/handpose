"""
TensorRT Engine 格式模型摄像头测试
====================================

使用 TensorRT Engine 格式的模型进行实时推理
"""

import cv2
import numpy as np
import time
import os
import platform
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mp_palmdet_trt import MPPalmDetTRT
from mp_handpose_trt import MPHandPoseTRT

# 检测操作系统
IS_WINDOWS = platform.system() == "Windows"

# 摄像头配置
USB_DEVICE_ID = 0
CAMERA_FRAMERATE = 30
DISPLAY_SIZE = (640, 640)

# 模型配置
PalmDet_MODEL = "palm_detection_lite_nchw"
HandLandmark_MODEL = "hand_landmark_lite_nchw"

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
        x1, y1, x2, y2 = palm[:4].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # 绘制手部结果
    for hand_results in hands:
        if hand_results is not None:
            # 绘制手部边界框
            hand_x1, hand_y1, hand_x2, hand_y2 = hand_results[:4].astype(int)
            cv2.rectangle(frame, (hand_x1, hand_y1), (hand_x2, hand_y2), (255, 0, 255), 2)
            
            # 绘制手部关键点
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
    print("=" * 80)
    print("TensorRT Engine 格式模型摄像头测试")
    print("=" * 80)
    
    # 模型路径
    palm_det_path = f"{PalmDet_MODEL}.engine"
    hand_landmark_path = f"{HandLandmark_MODEL}.engine"
    
    # 检查模型文件是否存在
    if not os.path.exists(palm_det_path):
        print(f"错误: 找不到模型文件 {palm_det_path}")
        return
    
    if not os.path.exists(hand_landmark_path):
        print(f"错误: 找不到模型文件 {hand_landmark_path}")
        return
    
    # 加载模型 - 使用 TensorRT 模型类
    print("\n加载模型...")
    try:
        palm_detector = MPPalmDetTRT(palm_det_path, use_gpu=True, scoreThreshold=0.3)
        hand_detector = MPHandPoseTRT(hand_landmark_path, use_gpu=True, confThreshold=0.1)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 打开摄像头
    print(f"\n打开摄像头 (设备ID: {USB_DEVICE_ID})...")
    cap = cv2.VideoCapture(USB_DEVICE_ID)
    
    if not cap.isOpened():
        print(f"✗ 无法打开摄像头 (设备ID: {USB_DEVICE_ID})")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FRAMERATE)
    
    print("✓ 摄像头打开成功")
    print("\n按 'q' 键退出")
    print("=" * 80)
    
    frame_count = 0
    total_time = 0
    start_time = time.time()
    elapsed_time = 0
    fps = 0
    avg_inference_time = 0
    
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("✗ 无法读取帧")
                break
            
            # 记录推理时间
            inference_start = time.time()
            
            # 手掌检测
            palms = palm_detector.infer(frame)
            
            # 调试信息
            if frame_count % 30 == 0:
                print(f"帧 {frame_count}: 检测到 {len(palms) if palms is not None else 0} 个手掌")
                if palms is not None and len(palms) > 0:
                    print(f"  第一个手掌 bbox: {palms[0][:4]}, score: {palms[0][18]:.4f}")
            
            # 手部关键点检测
            hands = []
            if palms is not None and len(palms) > 0:
                for i, palm in enumerate(palms):
                    handpose = hand_detector.infer(frame, palm)
                    if handpose is not None:
                        hands.append(handpose)
                        if frame_count % 30 == 0:
                            print(f"  手{i}检测成功，置信度: {handpose[131]:.4f}")
                    else:
                        if frame_count % 30 == 0:
                            print(f"  手{i}检测失败")
            
            inference_time = time.time() - inference_start
            total_time += inference_time
            frame_count += 1
            
            # 处理检测结果并绘制
            process_detection_results(frame, palms, hands)
            
            # 显示FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            avg_inference_time = total_time / frame_count if frame_count > 0 else 0
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Inference: {avg_inference_time*1000:.1f}ms", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Palms: {len(palms) if palms is not None else 0}, Hands: {len(hands)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "TensorRT Mode", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示帧
            cv2.imshow("TensorRT Camera Test", frame)
            
            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户退出")
                break
    
    except KeyboardInterrupt:
        print("\n程序被中断")
    
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 打印统计信息
        print("\n" + "=" * 80)
        print("统计信息:")
        print(f"总帧数: {frame_count}")
        print(f"总时间: {elapsed_time:.2f}s")
        print(f"平均FPS: {fps:.1f}")
        print(f"平均推理时间: {avg_inference_time*1000:.1f}ms")
        print("=" * 80)

if __name__ == "__main__":
    main()
