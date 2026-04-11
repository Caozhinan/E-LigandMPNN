import os
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 配置日志标准输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def download_cif(pdb_id: str, save_dir: Path, assembly: bool = True) -> bool:
    """
    下载指定 PDB ID 的 cif 文件。
    
    Args:
        pdb_id: 4位 PDB 编号
        save_dir: 存储目录的 Path 对象
        assembly: 是否下载 biological assembly 1. 若为 False，则下载 asymmetric unit.
    Returns:
        bool: 下载是否成功
    """
    pdb_id = pdb_id.lower()
    
    if assembly:
        filename = f"{pdb_id}-assembly1.cif"
        # 结构生物学常用 RCSB Assembly 节点
        url = f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif"
    else:
        filename = f"{pdb_id}.cif"
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        
    file_path = save_dir / filename
    
    # 避免重复下载
    if file_path.exists() and file_path.stat().st_size > 0:
        logging.info(f"File already exists: {filename}, skipping.")
        return True

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response, open(file_path, 'wb') as out_file:
            out_file.write(response.read())
        logging.info(f"Successfully downloaded: {filename}")
        return True
        
    except urllib.error.HTTPError as e:
        logging.error(f"HTTP Error for {filename}: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        logging.error(f"URL Error for {filename}: {e.reason}")
    except Exception as e:
        logging.error(f"Unexpected Error downloading {filename}: {str(e)}")
        
    # 如果下载失败但创建了空文件，则清理
    if file_path.exists():
        file_path.unlink()
    return False

def batch_download_cifs(pdb_list: list, output_dir: str, assembly: bool = True, max_workers: int = 4):
    """
    多线程并发下载 PDB CIF 文件
    """
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdb = {
            executor.submit(download_cif, pid, save_path, assembly): pid 
            for pid in pdb_list
        }
        
        for future in as_completed(future_to_pdb):
            if future.result():
                success_count += 1

    logging.info(f"Download complete. {success_count}/{len(pdb_list)} files downloaded successfully to {save_path.absolute()}.")

if __name__ == "__main__":
    # 测试集定义
    test_pdbs = [
        "1UBQ", "2LZM", # Monomers
        "1TIM", "4ZIN", # Homo-multimers
        "1BRS", "4HHB", # Hetero-multimers
        "1TSR", "1KX5", # Protein-DNA
        "1URN", "1A9N", # Protein-RNA
        "1STP", "1KE6", # Protein-Small Molecule
        "2CBA", "1CLL"  # Protein-Metal Ion
    ]
    
    # 默认下载 assembly1.cif 以符合物理相互作用逻辑
    batch_download_cifs(
        pdb_list=test_pdbs, 
        output_dir="./cif_test_data", 
        assembly=True, 
        max_workers=5
    )