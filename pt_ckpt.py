import torch
import sys
import os

def copy_pt_weights_into_ckpt(ckpt_path, pt_path, output_ckpt_path):
    """
    将 .pt 中的 model_state_dict 拷贝到 Lightning .ckpt 的 state_dict 中

    参数：
        ckpt_path: 原始 Lightning checkpoint
        pt_path:   包含 model_state_dict 的 .pt 文件
        output_ckpt_path: 输出的新 .ckpt 文件
    """
    try:
        # 加载 ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" not in ckpt:
            raise KeyError("ckpt 中未找到 'state_dict'")

        # 加载 pt
        pt = torch.load(pt_path, map_location="cpu")
        if "model_state_dict" not in pt:
            raise KeyError("pt 中未找到 'model_state_dict'")

        pt_state = pt["model_state_dict"]
        ckpt_state = ckpt["state_dict"]

        new_state_dict = {}
        replaced, skipped = 0, 0

        for k, v in ckpt_state.items():
            if k.startswith("model."):
                bare_key = k[len("model."):]
                if bare_key in pt_state:
                    new_state_dict[k] = pt_state[bare_key]
                    replaced += 1
                else:
                    new_state_dict[k] = v
                    skipped += 1
                    print(f"[Warning] pt 中缺少参数: {bare_key}")
            else:
                # 非 model 参数（如 optimizer、scheduler 相关）
                new_state_dict[k] = v

        ckpt["state_dict"] = new_state_dict

        torch.save(ckpt, output_ckpt_path)

        print("✅ 权重拷贝完成")
        print(f"  替换参数数目: {replaced}")
        print(f"  未替换参数数目: {skipped}")
        print(f"  输出文件: {output_ckpt_path}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:")
        print(f"  python {sys.argv[0]} input.ckpt input.pt output.ckpt")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    pt_path = sys.argv[2]
    output_ckpt_path = sys.argv[3]

    if not os.path.isfile(ckpt_path):
        print(f"❌ ckpt 文件不存在: {ckpt_path}")
        sys.exit(1)

    if not os.path.isfile(pt_path):
        print(f"❌ pt 文件不存在: {pt_path}")
        sys.exit(1)

    copy_pt_weights_into_ckpt(ckpt_path, pt_path, output_ckpt_path)

