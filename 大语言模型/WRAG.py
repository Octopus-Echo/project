import ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS

class LlamaWebAssistant:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def web_search(self, query, num_results=3):
        """使用 DuckDuckGo 进行网页搜索"""
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]
        return [r['href'] for r in results] if results else []

    def load_and_process(self, url):
        """加载并处理网页内容"""
        loader = WebBaseLoader(url)
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        return [chunk.page_content for chunk in chunks]

    def generate_response(self, question):
        """使用 Llama3.2 生成基于 Web 搜索的回答"""
        print("正在搜索网络...")
        urls = self.web_search(question)

        if not urls:
            return "未能找到相关网页信息。"

        print(f"找到的网页：{urls[:3]}")  # 显示前 3 个链接

        context = ""
        for url in urls[:3]:  # 只处理前 3 个网页
            try:
                print(f"正在获取：{url}")
                chunks = self.load_and_process(url)
                context += "\n\n".join(chunks[:3]) + "\n\n"  # 每个网页取前 3 段
            except Exception as e:
                print(f"获取网页失败：{url}，错误：{e}")
                continue

        if not context:
            return "未能从网页提取有效信息。"

        prompt = f"""请基于以下网络信息回答问题：
        {context}

        问题：{question}
        要求：
        1. 只基于提供的网络信息回答
        2. 如果信息不足请注明
        3. 保持回答简洁准确
        4. 在最后列出参考的信息来源"""

        print("正在生成回答...")
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']


if __name__ == "__main__":
    assistant = LlamaWebAssistant()
    print("Llama3 Web 问答助手（输入 'exit' 退出）")

    while True:
        question = input("\n请输入您的问题：").strip()
        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            print("请输入有效问题！")
            continue

        answer = assistant.generate_response(question)
        print("\n回答：")
        print(answer)