import onnx
from onnx import helper, numpy_helper
import numpy as np
import os

def analyze_and_fix_onnx(onnx_path, output_path=None):
    if output_path is None:
        output_path = onnx_path.replace('.onnx', '_nchw.onnx')
    
    print(f"\n{'='*70}")
    print(f"处理: {onnx_path}")
    print(f"{'='*70}")
    
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # 分析输入
    input_tensor = graph.input[0]
    input_shape = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
    input_name = input_tensor.name
    
    print(f"\n原始输入:")
    print(f"  名称: {input_name}")
    print(f"  形状: {input_shape}")
    
    # 检查第一个节点
    first_node = graph.node[0]
    print(f"\n第一个节点: {first_node.op_type}")
    
    if first_node.op_type == "Transpose":
        print("  发现 Transpose 节点，需要移除...")
        
        # 获取Transpose的输入和输出名称
        transpose_input = first_node.input[0]
        transpose_output = first_node.output[0]
        perm = list(first_node.attribute[0].ints) if first_node.attribute else [0, 3, 1, 2]
        print(f"  Transpose: {transpose_input} -> {transpose_output}, perm={perm}")
        
        # 检查是否是 NHWC -> NCHW (perm=[0,3,1,2])
        if perm == [0, 3, 1, 2]:
            # 更新输入形状: NHWC -> NCHW
            new_shape = [input_shape[0], input_shape[3], input_shape[1], input_shape[2]]
            print(f"  新输入形状: {new_shape}")
            
            # 更新输入张量的形状
            input_tensor.type.tensor_type.shape.ClearField('dim')
            for dim_val in new_shape:
                dim = input_tensor.type.tensor_type.shape.dim.add()
                dim.dim_value = dim_val
            
            # 更新输入名称
            input_tensor.name = transpose_input
            
            # 更新后续节点的输入，将 transpose_output 替换为 transpose_input
            updated_count = 0
            for node in graph.node[1:]:
                for i, inp_name in enumerate(node.input):
                    if inp_name == transpose_output:
                        node.input[i] = transpose_input
                        updated_count += 1
            
            print(f"  更新了 {updated_count} 个节点的输入")
            
            # 移除 Transpose 节点
            graph.node.remove(first_node)
            print("  已移除 Transpose 节点")
        else:
            print(f"  未知的 Transpose 排列: {perm}")
            return None
    else:
        print("  无 Transpose 节点")
        # 手动转换 NHWC -> NCHW
        if len(input_shape) == 4 and input_shape[3] == 3:
            new_shape = [input_shape[0], input_shape[3], input_shape[1], input_shape[2]]
            print(f"  手动转换输入形状: {input_shape} -> {new_shape}")
            
            input_tensor.type.tensor_type.shape.ClearField('dim')
            for dim_val in new_shape:
                dim = input_tensor.type.tensor_type.shape.dim.add()
                dim.dim_value = dim_val
            
            # 转换所有卷积层的权重
            for node in graph.node:
                if node.op_type == "Conv":
                    for i, inp_name in enumerate(node.input):
                        for init in graph.initializer:
                            if init.name == inp_name:
                                weight = numpy_helper.to_array(init)
                                if len(weight.shape) == 4:
                                    # NHWC -> NCHW: transpose(0,3,1,2)
                                    weight_new = weight.transpose(0, 3, 1, 2)
                                    new_init = numpy_helper.from_array(weight_new, init.name)
                                    init.CopyFrom(new_init)
                                    print(f"    转换权重: {init.name} {weight.shape} -> {weight_new.shape}")
    
    # 保存修改后的模型
    onnx.save(model, output_path)
    print(f"\n保存: {output_path}")
    
    # 验证
    print("\n验证:")
    model2 = onnx.load(output_path)
    inp = model2.graph.input[0]
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  输入: {inp.name}, 形状: {shape}")
    if len(shape) >= 4 and shape[1] == 3:
        print(f"  格式: NCHW ✓")
    elif len(shape) >= 4 and shape[3] == 3:
        print(f"  格式: NHWC")
    else:
        print(f"  格式: 未知")
    print(f"  第一个节点: {model2.graph.node[0].op_type}")
    
    return output_path


def main():
    base_dir = r"D:\ai_projects\llm_sop\knowledge-assistant\palm_hand"
    
    models = [
        {
            "onnx": os.path.join(base_dir, "palm_detection_lite.onnx"),
            "output": os.path.join(base_dir, "palm_detection_lite_nchw.onnx")
        },
        {
            "onnx": os.path.join(base_dir, "hand_landmark_lite.onnx"),
            "output": os.path.join(base_dir, "hand_landmark_lite_nchw.onnx")
        }
    ]
    
    print("="*70)
    print("ONNX 模型格式转换 (NHWC -> NCHW)")
    print("="*70)
    
    for model in models:
        if not os.path.exists(model["onnx"]):
            print(f"文件不存在: {model['onnx']}")
            continue
        
        analyze_and_fix_onnx(model["onnx"], model["output"])
    
    print("\n" + "="*70)
    print("转换完成!")
    print("="*70)


if __name__ == "__main__":
    main()
