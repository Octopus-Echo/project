import pandas as pd
import numpy as np
from ollama import Client
from typing import List, Dict, Any

# 初始化Ollama客户端
client = Client(host='http://localhost:11434')  # 默认Ollama地址


# 虚构的电影数据集
def create_movie_dataset():
    data = {
        "title": ["星际穿越", "盗梦空间", "泰坦尼克号", "霸王别姬", "阿凡达",
                  "这个杀手不太冷", "肖申克的救赎", "千与千寻", "教父", "大话西游"],
        "director": ["克里斯托弗·诺兰", "克里斯托弗·诺兰", "詹姆斯·卡梅隆", "陈凯歌",
                     "詹姆斯·卡梅隆", "吕克·贝松", "弗兰克·德拉邦特", "宫崎骏",
                     "弗朗西斯·福特·科波拉", "刘镇伟"],
        "year": [2014, 2010, 1997, 1993, 2009, 1994, 1994, 2001, 1972, 1995],
        "genre": ["科幻", "悬疑", "爱情", "剧情", "科幻", "动作", "剧情", "动画", "犯罪", "喜剧"],
        "rating": [9.3, 9.0, 9.4, 9.6, 8.8, 9.4, 9.7, 9.3, 9.2, 9.2],
        "duration_min": [169, 148, 194, 171, 162, 110, 142, 125, 175, 95],
        "language": ["英语", "英语", "英语", "普通话", "英语", "英语", "英语", "日语", "英语", "粤语"],
        "country": ["美国", "美国", "美国", "中国", "美国", "法国", "美国", "日本", "美国", "中国香港"]
    }
    return pd.DataFrame(data)


# 表检索函数
def retrieve_from_table(df: pd.DataFrame, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    从数据表中检索与查询最相关的条目
    """
    # 简单实现：基于字符串匹配的检索
    results = []
    query = query.lower()

    for _, row in df.iterrows():
        score = 0
        for col in df.columns:
            if query in str(row[col]).lower():
                score += 1
        if score > 0:
            results.append((score, row.to_dict()))

    # 按相关性排序并返回前top_k个结果
    results.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in results[:top_k]]


# 使用Llama3.2生成增强回答
def generate_augmented_response(query: str, retrieved_data: List[Dict[str, Any]]) -> str:
    """
    使用Llama3.2模型结合检索到的数据生成增强回答
    """
    # 构建提示词
    context = "\n".join([str(item) for item in retrieved_data])
    prompt = f"""你是一个电影专家，请根据以下检索到的电影信息和你的知识回答用户问题。

检索到的信息：
{context}

用户问题：{query}

请给出专业、详细的回答，并适当引用检索到的信息："""

    # 调用Ollama的Llama3.2模型
    response = client.generate(
        model='llama3.2',
        prompt=prompt,
        options={
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 1000
        }
    )

    return response['response']


# 主函数
def main():
    # 加载数据
    movie_df = create_movie_dataset()

    # 示例查询
    query = "推荐一部中国的电影"

    # 1. 从表中检索相关信息
    retrieved_data = retrieve_from_table(movie_df, query)
    print("检索到的数据：")
    for item in retrieved_data:
        print(item)

    # 2. 使用Llama3.2生成增强回答
    response = generate_augmented_response(query, retrieved_data)
    print("\n增强回答：")
    print(response)


if __name__ == "__main__":
    main()