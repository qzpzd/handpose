import tensorrt as trt
import os
import pycuda.driver as cuda
import pycuda.autoinit


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def convert_onnx_to_engine(onnx_path, engine_path, fp16=True):
    """将 ONNX 模型转换为 TensorRT Engine"""
    print(f"\n{'='*70}")
    print(f"转换: {onnx_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(onnx_path):
        print(f"ONNX 文件不存在: {onnx_path}")
        return False
    
    if os.path.exists(engine_path):
        os.remove(engine_path)
        print(f"删除旧 engine: {engine_path}")
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ONNX 解析失败!")
            for i in range(parser.num_errors):
                print(f"  错误 {i}: {parser.get_error(i)}")
            return False
    
    print("ONNX 解析成功")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用 FP16")
    
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    print(f"输入形状: {input_shape}")
    
    if -1 in input_shape:
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_tensor.name,
            (1, input_shape[1], input_shape[2], input_shape[3]),
            (1, input_shape[1], input_shape[2], input_shape[3]),
            (1, input_shape[1], input_shape[2], input_shape[3])
        )
        config.add_optimization_profile(profile)
        print("设置动态 shape profile")
    
    print("构建 TensorRT engine...")
    engine_data = builder.build_serialized_network(network, config)
    
    if engine_data is None:
        print("TensorRT engine 构建失败!")
        return False
    
    with open(engine_path, "wb") as f:
        f.write(engine_data)
    
    file_size = os.path.getsize(engine_path) / 1024 / 1024
    print(f"保存: {engine_path} ({file_size:.2f} MB)")
    return True


def verify_engine(engine_path):
    """验证 TensorRT Engine 的输入输出"""
    print(f"\n{'='*70}")
    print(f"验证: {engine_path}")
    print(f"{'='*70}")
    
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("Engine 加载失败!")
        return False
    
    print("\n输入:")
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            print(f"  {i}: {name}, shape={shape}")
    
    print("\n输出:")
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            print(f"  {i}: {name}, shape={shape}")
    
    return True


def main():
    base_dir = r"D:\ai_projects\llm_sop\knowledge-assistant\palm_hand"
    
    models = [
        {
            "onnx": os.path.join(base_dir, "palm_detection_lite_nchw.onnx"),
            "engine": os.path.join(base_dir, "palm_detection_lite_nchw.engine")
        },
        {
            "onnx": os.path.join(base_dir, "hand_landmark_lite_nchw.onnx"),
            "engine": os.path.join(base_dir, "hand_landmark_lite_nchw.engine")
        }
    ]
    
    print("="*70)
    print("ONNX to TensorRT 转换")
    print("="*70)
    
    for model in models:
        if convert_onnx_to_engine(model["onnx"], model["engine"], fp16=True):
            verify_engine(model["engine"])
    
    print("\n" + "="*70)
    print("所有模型转换完成!")
    print("="*70)


if __name__ == "__main__":
    main()
