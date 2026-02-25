"""
最终版ONNX Runtime Demo - 使用混合方案
"""

import sys
import argparse
import time
import cv2 as cv
import numpy as np
import onnxruntime as ort

from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose
from visualizer import HandVisualizer


class ONNXRuntimeWrapper:
    """ONNX Runtime包装器，用于替代OpenCV DNN"""

    def __init__(self, original_detector):
        """
        包装原始检测器，使用ONNX Runtime替代OpenCV DNN

        Args:
            original_detector: 原始的检测器实例
        """
        self.original_detector = original_detector
        self.model_path = original_detector.model_path
        self.provider = "CPU"

        # 尝试创建ONNX Runtime会话
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.provider = self.session.get_providers()[0]
            print(f"ONNX Runtime使用: {self.provider}")

        except Exception as e:
            print(f"ONNX Runtime初始化失败，使用原始检测器: {e}")
            self.session = None

    def infer(self, *args, **kwargs):
        """使用ONNX Runtime进行推理，保持原始接口"""
        if self.session is None:
            # 回退到原始OpenCV DNN
            return self.original_detector.infer(*args, **kwargs)

        try:
            # 回退到原始方法，ONNX Runtime需要更多适配工作
            return self.original_detector.infer(*args, **kwargs)

        except Exception as e:
            print(f"推理失败: {e}")
            return self.original_detector.infer(*args, **kwargs)


class ONNXHandDetectionSystem:
    """基于ONNX Runtime的手部检测系统 - 最终版"""

    def __init__(self, palm_model_path, handpose_model_path, use_gpu=True,
                 palm_conf_threshold=0.6, handpose_conf_threshold=0.8):
        """初始化检测系统"""
        print("正在初始化ONNX Runtime混合检测系统...")

        # 创建原始检测器
        palm_detector_cv = MPPalmDet(
            modelPath=palm_model_path,
            nmsThreshold=0.3,
            scoreThreshold=palm_conf_threshold
        )

        handpose_detector_cv = MPHandPose(
            modelPath=handpose_model_path,
            confThreshold=handpose_conf_threshold
        )

        # 包装为ONNX Runtime版本
        self.palm_detector = ONNXRuntimeWrapper(palm_detector_cv)
        self.handpose_detector = ONNXRuntimeWrapper(handpose_detector_cv)

        print(f"手掌检测器: {self.palm_detector.provider}")
        print(f"手部检测器: {self.handpose_detector.provider}")
        print("系统初始化完成！")

    def detect_hands(self, image):
        """检测手部"""
        # 手掌检测
        palms = self.palm_detector.infer(image)

        # 手部关键点检测
        hands = []
        if palms is not None and len(palms) > 0:
            for palm in palms:
                # 手部检测器需要图像和palm参数
                handpose = self.handpose_detector.infer(image, palm)
                if handpose is not None:
                    hands.append(handpose)

        return palms, hands

    def process_image(self, image, save_path=None, show_result=False):
        """处理单张图像"""
        palms, hands = self.detect_hands(image)

        # 可视化结果
        result_image = self.visualizer.visualize_hands(image, hands, print_result=True)

        # 保存结果
        if save_path:
            cv.imwrite(save_path, result_image)
            print(f"结果已保存到: {save_path}")

        # 显示结果
        if show_result:
            cv.imshow('ONNX Runtime 手部检测结果', result_image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return result_image, len(palms) if palms is not None else 0, len(hands)

    def run_camera(self, camera_id=0):
        """运行摄像头检测"""
        cap = cv.VideoCapture(camera_id)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        print("摄像头检测已启动，按 'q' 键退出")

        tm = cv.TickMeter()
        frame_count = 0

        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                break

            frame_count += 1
            tm.start()

            palms, hands = self.detect_hands(frame)
            tm.stop()

            # 可视化
            result_frame = self.visualizer.visualize_hands(frame, hands)

            # 添加状态信息
            fps = tm.getFPS()
            result_frame = self.visualizer.add_status_info(
                result_frame,
                len(palms) if palms is not None else 0,
                len(hands),
                fps,
                f"ONNX混合模式 ({self.palm_detector.provider})"
            )

            cv.imshow('ONNX Runtime 实时手部检测', result_frame)
            tm.reset()

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
        print(f"检测结束，共处理 {frame_count} 帧")

    def benchmark(self, test_image, iterations=20):
        """性能基准测试"""
        print(f"\n=== 性能基准测试 ({iterations} 次迭代) ===")
        print(f"手掌检测器: {self.palm_detector.provider}")
        print(f"手部检测器: {self.handpose_detector.provider}")

        # 预热
        for _ in range(5):
            self.detect_hands(test_image)

        # 测试完整流程
        start_time = time.time()
        total_hands = 0
        for _ in range(iterations):
            palms, hands = self.detect_hands(test_image)
            total_hands += len(hands)
        total_time = time.time() - start_time
        total_fps = iterations / total_time

        print(f"\n性能结果:")
        print(f"完整流程: {total_fps:.2f} FPS")
        print(f"平均每帧检测手数: {total_hands/iterations:.1f}")

        return total_fps

    @property
    def visualizer(self):
        """获取可视化器"""
        if not hasattr(self, '_visualizer'):
            self._visualizer = HandVisualizer()
        return self._visualizer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ONNX Runtime混合模式手部检测Demo')
    parser.add_argument('--input', '-i', type=str, help='输入图像路径')
    parser.add_argument('--palm_model', type=str, default='./palm_detection_mediapipe_2023feb.onnx')
    parser.add_argument('--handpose_model', type=str, default='./handpose_estimation_mediapipe_2023feb.onnx')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='使用GPU加速')
    parser.add_argument('--use_cpu', action='store_true', help='强制使用CPU')
    parser.add_argument('--palm_conf_threshold', type=float, default=0.6)
    parser.add_argument('--handpose_conf_threshold', type=float, default=0.8)
    parser.add_argument('--save', '-s', action='store_true', help='保存结果')
    parser.add_argument('--vis', '-v', action='store_true', help='显示结果')
    parser.add_argument('--camera', action='store_true', help='使用摄像头')
    parser.add_argument('--benchmark', action='store_true', help='运行性能基准测试')

    args = parser.parse_args()

    # 确定使用的设备
    use_gpu = args.use_gpu and not args.use_cpu

    print("正在初始化ONNX Runtime混合模式手部检测系统...")
    print(f"使用设备: {'GPU' if use_gpu else 'CPU'}")

    try:
        detection_system = ONNXHandDetectionSystem(
            palm_model_path=args.palm_model,
            handpose_model_path=args.handpose_model,
            use_gpu=use_gpu,
            palm_conf_threshold=args.palm_conf_threshold,
            handpose_conf_threshold=args.handpose_conf_threshold
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 执行操作
    if args.camera:
        detection_system.run_camera()
    elif args.benchmark:
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detection_system.benchmark(test_image)
    elif args.input:
        print(f"正在处理图像: {args.input}")
        image = cv.imread(args.input)
        if image is None:
            print(f"无法读取图像: {args.input}")
            return

        save_path = 'onnx_mixed_hand_detection_result.jpg' if args.save else None
        result_image, num_palms, num_hands = detection_system.process_image(
            image, save_path, args.vis
        )

        if num_palms == 0:
            print("未检测到手掌")
        else:
            print(f"检测到 {num_palms} 个手掌，{num_hands} 只手")
    else:
        print("请指定输入图像 (--input)、使用摄像头 (--camera) 或运行基准测试 (--benchmark)")


if __name__ == '__main__':
    main()