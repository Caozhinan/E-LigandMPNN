import torch
import sys
import os

def convert_ckpt_to_pt(ckpt_path, output_path):
    """
    转换Lightning CKPT文件为定制格式的PT参数文件
    参数：
        ckpt_path: 输入的.ckpt文件路径
        output_path: 输出的.pt文件路径
    """
    try:
        # 加载checkpoint文件
        ckpt_weight = torch.load(ckpt_path, map_location='cpu')
        
        # 从参考模型文件加载模板参数
        model_weight = torch.load("/xcfhome/ypxia/Workspace/LigandMPNN/model_params/publication_version_ligandmpnn_v_32_010_25.pt", map_location='cpu')

        # 构建新的模型参数字典（使用参考模型的元参数）
        new_model = {
            'num_edges': model_weight['num_edges'],
            'noise_level': model_weight['noise_level'],
            'atom_context_num': 30  # 更新为新值
        }

        # 处理state_dict键名（从ckpt提取并去除前缀）
        model_state_dict = {}
        for key in ckpt_weight['state_dict']:
            if key.startswith('model.'):  # 严格匹配需要转换的键
                model_state_dict[key[6:]] = ckpt_weight['state_dict'][key]
            else:
                print(f"[Warning] Skipped unexpected key: {key}")

        new_model['model_state_dict'] = model_state_dict

        # 确保输出目录存在
        #os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存合并后的参数
        torch.save(new_model, output_path)
        print(f"✅ Conversion successful! Saved to: {output_path}")

    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 参数验证
    if len(sys.argv) != 3:
        print("Usage:")
        print(f"  python {sys.argv[0]} [input.ckpt_path] [output.pt_path]")
        print("\nExample:")
        print(f"  python {sys.argv[0]} ./input.ckpt ./output/model.pt")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # 输入文件存在性检查
    if not os.path.isfile(input_path):
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    # 执行转换流程
    convert_ckpt_to_pt(input_path, output_path)
