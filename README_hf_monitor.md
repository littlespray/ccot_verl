# Hugging Face 模型上传和监控工具

本工具用于将本地目录自动上传到 Hugging Face 模型仓库，并持续监控目录变化，自动更新模型。

## 功能特点

- 🚀 将指定目录上传到 Hugging Face 模型仓库
- ⏰ 每小时自动检查目录变化并更新
- 📦 自动管理模型版本，最多保留 5 个最新版本
- 🛑 提供停止脚本，可以优雅地终止监控程序
- 📝 保存状态信息，支持程序重启后恢复

## 安装依赖

```bash
pip install -r requirements_hf_monitor.txt
```

## 使用方法

### 1. 启动监控程序

```bash
python upload_and_monitor.py <输入目录> <HF_TOKEN>
```

参数说明：
- `<输入目录>`: 要上传和监控的本地目录路径
- `<HF_TOKEN>`: 你的 Hugging Face API token（可以从 https://huggingface.co/settings/tokens 获取）

示例：
```bash
python upload_and_monitor.py ./my_model_dir hf_abcdefghijklmnopqrstuvwxyz
```

可选参数：
- `--username`: Hugging Face 用户名（默认为 sunshk）

### 2. 停止监控程序

```bash
python stop_monitor.py
```

该脚本会：
1. 查找运行中的监控进程
2. 发送停止信号
3. 等待进程优雅退出
4. 可选择是否删除状态文件

### 3. 查看状态

监控程序会创建以下文件：
- `hf_monitor.pid`: 存储进程 ID
- `hf_monitor_state.json`: 保存已上传的模型列表和最后检查时间

## 工作原理

1. **目录哈希计算**: 基于文件内容、修改时间和大小计算目录的唯一哈希值
2. **变化检测**: 每小时计算一次哈希值，如果与上次不同则触发上传
3. **模型命名**: 使用 `目录名_YYYYMMDD_HHMMSS` 格式命名模型
4. **版本管理**: 自动删除最旧的模型，保持最多 5 个版本
5. **信号处理**: 支持 Ctrl+C 和 SIGTERM 信号优雅退出

## 注意事项

1. **Token 安全**: 请妥善保管你的 Hugging Face token，不要将其提交到版本控制系统
2. **网络要求**: 需要稳定的网络连接来上传大文件
3. **存储空间**: Hugging Face 免费账户有存储限制，请注意模型大小
4. **权限要求**: 确保对输入目录有读取权限

## 常见问题

### Q: 如何在后台运行监控程序？
A: 使用 `nohup` 或 `screen`：
```bash
nohup python upload_and_monitor.py ./my_model_dir hf_token > monitor.log 2>&1 &
```

### Q: 如何修改检查间隔？
A: 编辑 `upload_and_monitor.py` 中的 `check_interval` 变量（默认 3600 秒）

### Q: 如何恢复中断的监控？
A: 重新运行启动命令即可，程序会读取状态文件并继续监控

### Q: 如何查看已上传的模型？
A: 查看 `hf_monitor_state.json` 文件或访问 https://huggingface.co/sunshk

## 故障排除

1. **上传失败**: 检查网络连接和 token 权限
2. **目录无权限**: 确保对输入目录有读取权限
3. **进程无法停止**: 使用 `kill -9 <PID>` 强制终止

## 许可证

本工具遵循项目主许可证。
