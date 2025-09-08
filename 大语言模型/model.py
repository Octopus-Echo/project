import requests
def ask_llama3(question):
    # Ollama 的本地 API 地址
    url = "http://localhost:11434/api/generate"

    # 请求数据（指定模型和问题）
    data = {
        "model": "llama3.2",
        "prompt": question,
        "stream": False  # 设置为 False 直接获取完整响应
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # 检查请求是否成功
        return response.json()["response"]
    except Exception as e:
        return f"Error: {str(e)}"
# 示例使用
question = "为什么天空是蓝色的？"
answer = ask_llama3(question)
print("回答:", answer)