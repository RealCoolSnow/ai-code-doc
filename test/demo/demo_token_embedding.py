import sys
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError:
    print("需要安装 transformers 和 torch 库。请运行: pip install transformers torch")
    sys.exit(1)

def main():
    # 加载模型和分词器
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"正在加载模型: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 用户输入
    user_input = input("请输入一个字符串: ")
    # Tokenize
    tokens = tokenizer(user_input, return_tensors="pt")
    token_length = len(tokens["input_ids"][0])
    print(f"Token 长度: {token_length}")

    # 获取嵌入向量
    with torch.no_grad():
        output = model(**tokens)
        # 取 [CLS] token 的输出作为句子向量
        embeddings = output.last_hidden_state[:, 0, :].squeeze().numpy()
    print(f"嵌入向量 (前10维): {embeddings[:10]}")
    print(f"嵌入向量总长度: {embeddings.shape[0]}")

if __name__ == "__main__":
    main() 