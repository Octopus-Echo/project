import pandas as pd
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# 虚构电影数据集
def create_movie_dataset():
    movies = [
        {
            "title": "最后的冒险",
            "year": 2023,
            "genre": "动作, 冒险",
            "director": "艾米丽·陈",
            "actors": "克里斯·帕拉特, 佐伊·索尔达娜, 戴夫·巴蒂斯塔",
            "rating": 8.7,
            "plot": "一群探险家穿越银河系寻找一个神话般的星球。"
        },
        {
            "title": "午夜巴黎",
            "year": 2011,
            "genre": "爱情, 奇幻",
            "director": "伍迪·艾伦",
            "actors": "欧文·威尔逊, 瑞秋·麦克亚当斯, 玛丽昂·歌迪亚",
            "rating": 7.7,
            "plot": "一位编剧发现自己每天午夜都会回到1920年代。"
        },
        {
            "title": "量子悖论",
            "year": 2022,
            "genre": "科幻, 惊悚",
            "director": "克里斯托弗·诺兰",
            "actors": "约翰·大卫·华盛顿, 罗伯特·帕丁森",
            "rating": 8.5,
            "plot": "一位物理学家发现了与平行宇宙交流的方法。"
        }
    ]
    return pd.DataFrame(movies)


# 文本处理工具
class TextProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.is_fitted = False

    def fit_transform(self, texts: List[str]):
        embeddings = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return embeddings

    def transform(self, texts: List[str]):
        if not self.is_fitted:
            raise ValueError("Vectorizer尚未训练，请先调用fit_transform方法")
        return self.vectorizer.transform(texts)


# 电影检索系统
class MovieRetriever:
    def __init__(self):
        self.movie_df = create_movie_dataset()
        self.text_processor = TextProcessor()
        self._prepare_embeddings()

    def _prepare_embeddings(self):
        # 合并电影的各种文本信息用于向量化
        combined_texts = self.movie_df.apply(
            lambda x: f"{x['title']} {x['plot']} {x['genre']} {x.get('director', '')}",
            axis=1
        ).tolist()
        self.embeddings = self.text_processor.fit_transform(combined_texts)

    def retrieve(self, query: str, top_k: int = 3) -> pd.DataFrame:
        # 将查询文本向量化
        query_embedding = self.text_processor.transform([query])

        # 计算余弦相似度，确保结果是一维数组
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()

        # 创建结果DataFrame并排序
        result_df = self.movie_df.copy()
        result_df['similarity'] = similarities
        return result_df.sort_values('similarity', ascending=False).head(top_k)


# 响应生成器
class ResponseGenerator:
    @staticmethod
    def generate(query: str, movies: pd.DataFrame) -> str:
        context = "\n".join([
            f"- {row['title']} ({row['year']}): {row['plot']} [评分: {row['rating']}]"
            for _, row in movies.iterrows()
        ])

        return f"""根据您的查询"{query}"，我推荐以下电影：

{context}

这些电影应该符合您的兴趣。您想了解哪部的更多信息？"""


# 主系统
class MovieRecommendationSystem:
    def __init__(self):
        self.retriever = MovieRetriever()
        self.generator = ResponseGenerator()

    def query(self, user_query: str) -> Dict:
        relevant_movies = self.retriever.retrieve(user_query)
        response = self.generator.generate(user_query, relevant_movies)

        return {
            "movies": relevant_movies[['title', 'year', 'genre', 'rating', 'similarity']].to_dict('records'),
            "response": response
        }


# 使用示例
if __name__ == "__main__":
    system = MovieRecommendationSystem()

    while True:
        user_input = input("\n请输入您的电影查询（或输入'退出'结束）: ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            break

        result = system.query(user_input)

        print("\n=== 推荐电影 ===")
        for movie in result['movies']:
            print(f"{movie['title']} ({movie['year']}) - 类型: {movie['genre']} - 相似度: {movie['similarity']:.2f}")

        print("\n=== 推荐理由 ===")
        print(result['response'])