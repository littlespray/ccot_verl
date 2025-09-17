#!/usr/bin/env python3
"""
Hugging Face 模型上传和监控脚本
- 将指定目录上传到 Hugging Face 模型仓库
- 每小时检查目录更新，自动上传新版本
- 最多保持 5 个最近的模型版本
"""

import os
import sys
import time
import argparse
import signal
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from huggingface_hub import HfApi, create_repo, upload_folder


class HuggingFaceModelMonitor:
    def __init__(self, input_dir: str, hf_token: str, username: str = "sunshk"):
        self.input_dir = Path(input_dir).resolve()
        self.hf_token = hf_token
        self.username = username
        self.api = HfApi(token=hf_token)
        self.pid_file = Path("hf_monitor.pid")
        self.state_file = Path("hf_monitor_state.json")
        self.running = True
        self.models: List[str] = []
        
        # 验证输入目录
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        # 设置信号处理
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # 保存 PID
        self._save_pid()
        
        # 加载状态
        self._load_state()
    
    def _signal_handler(self, signum, frame):
        """处理终止信号"""
        print(f"\nReceived signal {signum}, stopping monitor...")
        self.running = False
    
    def _save_pid(self):
        """保存进程 ID"""
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
    
    def _cleanup(self):
        """清理 PID 文件"""
        if self.pid_file.exists():
            self.pid_file.unlink()
    
    def _save_state(self):
        """保存状态到文件"""
        state = {
            "models": self.models,
            "last_check": datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """从文件加载状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.models = state.get("models", [])
            except Exception as e:
                print(f"Failed to load state: {e}")
                self.models = []
    
    def _calculate_directory_hash(self) -> str:
        """计算目录的哈希值（基于文件内容和修改时间）"""
        hash_obj = hashlib.sha256()
        
        for root, dirs, files in os.walk(self.input_dir):
            # 排序以确保一致性
            dirs.sort()
            files.sort()
            
            for filename in files:
                filepath = Path(root) / filename
                # 跳过隐藏文件和临时文件
                if filename.startswith('.') or filename.endswith('~'):
                    continue
                
                try:
                    # 添加文件路径（相对路径）
                    rel_path = filepath.relative_to(self.input_dir)
                    hash_obj.update(str(rel_path).encode())
                    
                    # 添加文件修改时间
                    mtime = filepath.stat().st_mtime
                    hash_obj.update(str(mtime).encode())
                    
                    # 添加文件大小
                    size = filepath.stat().st_size
                    hash_obj.update(str(size).encode())
                    
                    # 对于小文件，添加内容哈希
                    if size < 10 * 1024 * 1024:  # 10MB
                        with open(filepath, 'rb') as f:
                            hash_obj.update(f.read())
                
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
        
        return hash_obj.hexdigest()
    
    def _generate_model_name(self) -> str:
        """生成模型名称（目录名 + 时间戳）"""
        dir_name = self.input_dir.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{dir_name}_{timestamp}"
    
    def _upload_to_huggingface(self, model_name: str) -> bool:
        """上传目录到 Hugging Face"""
        try:
            repo_id = f"{self.username}/{model_name}"
            
            print(f"Creating repository: {repo_id}")
            # 创建仓库（如果不存在）
            create_repo(
                repo_id=repo_id,
                token=self.hf_token,
                private=False,
                exist_ok=True
            )
            
            print(f"Uploading directory {self.input_dir} to {repo_id}")
            # 上传整个目录
            upload_folder(
                folder_path=str(self.input_dir),
                repo_id=repo_id,
                token=self.hf_token,
                commit_message=f"Upload model at {datetime.now().isoformat()}"
            )
            
            print(f"Successfully uploaded to: https://huggingface.co/{repo_id}")
            return True
            
        except Exception as e:
            print(f"Upload failed: {e}")
            return False
    
    def _cleanup_old_models(self):
        """删除旧的模型，保持最多 5 个"""
        if len(self.models) > 5:
            # 获取需要删除的模型
            models_to_delete = self.models[:-5]
            
            for model_name in models_to_delete:
                try:
                    repo_id = f"{self.username}/{model_name}"
                    print(f"Deleting old model: {repo_id}")
                    self.api.delete_repo(
                        repo_id=repo_id,
                        token=self.hf_token,
                        repo_type="model"
                    )
                    print(f"Deleted: {repo_id}")
                except Exception as e:
                    print(f"Failed to delete model {repo_id}: {e}")
            
            # 更新模型列表
            self.models = self.models[-5:]
            self._save_state()
    
    def run(self):
        """运行监控程序"""
        print(f"Starting to monitor directory: {self.input_dir}")
        print(f"Process ID: {os.getpid()}")
        print("Press Ctrl+C or use stop_monitor.py to stop monitoring\n")
        
        last_hash = ""
        check_interval = 3600  # 1 小时
        
        try:
            while self.running:
                # 计算当前目录哈希
                current_hash = self._calculate_directory_hash()
                
                # 检查是否有变化
                if current_hash != last_hash:
                    print(f"\n[{datetime.now()}] Directory changes detected")
                    
                    # 生成新的模型名称
                    model_name = self._generate_model_name()
                    
                    # 上传到 Hugging Face
                    if self._upload_to_huggingface(model_name):
                        self.models.append(model_name)
                        self._save_state()
                        
                        # 清理旧模型
                        self._cleanup_old_models()
                        
                        last_hash = current_hash
                    else:
                        print("Upload failed, will retry on next check")
                else:
                    print(f"[{datetime.now()}] No directory changes, skipping upload")
                
                if self.running:
                    print(f"Waiting {check_interval} seconds before next check...")
                    # 使用短暂的睡眠循环，以便能够响应信号
                    for _ in range(check_interval):
                        if not self.running:
                            break
                        time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt")
        
        finally:
            print("Cleaning up...")
            self._cleanup()
            print("Monitor program stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Upload directory to Hugging Face and monitor updates"
    )
    parser.add_argument(
        "--input_dir",
        help="Input directory to upload and monitor"
    )
    parser.add_argument(
        "--hf_token",
        help="Hugging Face API token"
    )
    parser.add_argument(
        "--username",
        default="sunshk",
        help="Hugging Face username (default: sunshk)"
    )
    
    args = parser.parse_args()
    
    try:
        monitor = HuggingFaceModelMonitor(
            input_dir=args.input_dir,
            hf_token=args.hf_token,
            username=args.username
        )
        monitor.run()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
