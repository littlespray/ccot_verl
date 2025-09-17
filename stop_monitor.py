#!/usr/bin/env python3
"""
停止 Hugging Face 模型监控程序
"""

import os
import sys
import signal
import time
from pathlib import Path


def stop_monitor():
    """停止监控程序"""
    pid_file = Path("hf_monitor.pid")
    
    # 检查 PID 文件是否存在
    if not pid_file.exists():
        print("未找到运行中的监控程序（PID 文件不存在）")
        return False
    
    try:
        # 读取 PID
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        print(f"找到监控进程，PID: {pid}")
        
        # 检查进程是否存在
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print("监控进程已经不存在")
            pid_file.unlink()
            return False
        
        # 发送 SIGTERM 信号
        print("正在发送停止信号...")
        os.kill(pid, signal.SIGTERM)
        
        # 等待进程结束（最多等待 10 秒）
        for i in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(1)
                print(f"等待进程停止... ({i+1}/10)")
            except ProcessLookupError:
                print("监控程序已成功停止")
                return True
        
        # 如果进程仍在运行，尝试强制终止
        print("进程未响应 SIGTERM，尝试强制终止...")
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
            print("监控程序已被强制终止")
        except ProcessLookupError:
            print("监控程序已停止")
        
        return True
        
    except Exception as e:
        print(f"停止监控程序时出错: {e}")
        return False


def main():
    print("=== Hugging Face Uploading Finished ===\n")
    stop_monitor()


if __name__ == "__main__":
    main()
