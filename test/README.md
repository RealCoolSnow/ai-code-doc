# test 目录说明

## demo_token_embedding.py

此脚本演示如何提示用户输入字符串，计算其 token 长度，并将其转换为向量（使用 HuggingFace sentence-transformers/all-MiniLM-L6-v2 模型）。

### 使用方法

1. 安装依赖：
   ```bash
   pip install transformers torch
   ```
2. 运行脚本：
   ```bash
   python demo_token_embedding.py
   ```
3. 按提示输入字符串，脚本会输出 token 长度和嵌入向量的前 10 维。
