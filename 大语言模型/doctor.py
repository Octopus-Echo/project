import chromadb
from chromadb.utils import embedding_functions
import re
import ollama  # 确保ollama库已正确安装

# 初始化chromadb客户端
client = chromadb.Client()

# 初始化ollama客户端
ollama_client = ollama.Client()

# 使用Ollama嵌入函数
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
   model_name="nomic-embed-text:latest",
   url="http://localhost:11434/api/embeddings",
)

# 创建舌象知识库collection
collection = client.get_or_create_collection(
   name="tongue_diagnosis_collection",
   embedding_function=ollama_ef
)

# 中医舌象知识库文档
tongue_knowledge = [
   "舌象包括舌质和舌苔变化，舌为肌性器官，由黏膜和舌肌组成。",
   "舌的主要功能是辨别滋味，调节声音，拌合食物，协助吞咽。",
   "舌由肌肉、血脉和经络所构成，三者都与脏腑存在着密切的联系。",
   "中医看病讲究望闻问切，更是看重舌苔的改变。透过舌象的观察来了解和认识疾病的本质和发展。",
   "胃病的治疗也是，舌象变化确实能反应一些脾胃病的规律。",
   "舌象变化规律一般来说为舌苔由薄变厚为病进，由厚变薄为病退。",
   "伸舌时要自然，舌体放松，舌面平展，舌尖略向下，口尽量张大。",
   "望舌要有次序，一般先看舌尖，再看舌中、舌侧，最后看舌根部。",
   "舌诊主要观察舌体和舌苔两个方面的变化。",
   "舌体观察要点：颜色、形状(胖瘦)、质地(荣枯)、活动是否灵活自如。",
   "舌苔观察要点：苔质(厚薄、润燥)、苔色(白、黄、棕褐等)。",
   "正常舌象特征：淡红舌薄白苔，舌色淡红鲜明，舌质滋润，舌体大小适中，柔软灵活。",
   "舌质淡白多主虚证、寒证，舌质红绛多主热证。",
   "舌苔白厚腻多为寒湿或食积，舌苔黄腻多为湿热或痰热。",
   "舌体胖大有齿痕多属脾虚湿盛，舌体瘦薄多属气血不足。",
   "裂纹舌多见于阴血亏虚，芒刺舌多见于热邪亢盛。",
   "舌下络脉青紫怒张多提示血瘀证候。"
]

# 分割文本为句子
def split_regex_sentence(text_list):
   chunks = []
   for i, text in enumerate(text_list):
       # 移除特殊字符
       text = re.sub(r'[\r\n\t]', ' ', text)
       sentences = re.split(r'(?<=[。！？])', text)
       chunks.extend([(f"doc{i}_s{j}", s.strip())
                     for j, s in enumerate(sentences) if s.strip()])
   return chunks

# 添加文档到向量数据库
def add2vecdatabase(collection_name, documents, ids):
   collection = client.get_or_create_collection(name=collection_name, embedding_function=ollama_ef)
   collection.add(
       ids=ids,
       documents=documents
   )
   return collection

# 查询向量数据库
def query_collection(collection_name, query_text, topk=3):
   collection = client.get_or_create_collection(name=collection_name, embedding_function=ollama_ef)
   result = collection.query(
       query_texts=[query_text],
       n_results=topk
   )
   return result

# 修改后的LLM对话函数
def llmChat(prompt, systemStr):
   try:
       response = ollama_client.chat(
           model="llama3.2",  # 确保模型名称与本地安装的ollama模型一致
           messages=[
               {'role': 'system', 'content': systemStr},
               {"role": "user", "content": prompt}
           ]
       )
       return response['message']['content']
   except Exception as e:
       return f"调用模型出错: {str(e)}"

# 中医舌象问诊函数
def tongue_diagnosis_qa(question):
   # 系统角色设定
   system_role = """您是一位资深中医专家，精通舌诊。请根据以下原则回答问题：
   1. 诊断依据：严格基于提供的中医舌象知识
   2. 回答结构：
      - 直接结论
      - 理论解释（引用经典理论）
      - 实用建议（包括饮食、生活习惯等）
   3. 注意事项：
      - 对不确定的情况明确说明
      - 避免绝对化表述
      - 保持专业且亲切的语气"""

   # 从知识库检索相关文档
   result = query_collection("tongue_diagnosis_collection", question, topk=5)
   context = " ".join(result['documents'][0])
   enhanced_query = f"{question}\n相关背景知识：{context}"

   # 获取LLM的回答
   response = llmChat(enhanced_query, system_role)
   return response

if __name__ == "__main__":
   print("=== 中医舌象问诊系统 ===")
   print("系统初始化检查：")

   # 环境检查
   try:
       import doctor
       print("- Ollama库: 已安装")
   except ImportError:
       print("错误: Ollama库未安装，请运行: pip install ollama")
       exit()

   print("\n可以开始咨询了（输入'退出'结束）")
   while True:
       question = input("\n患者咨询：")
       if question.lower() in ['退出', 'exit', 'quit']:
           break
       print("\n中医专家思考中...")
       answer = tongue_diagnosis_qa(question)
       print("\n回答：" + answer)