import re
from collections import defaultdict


class OptimizedChineseQA:
    def __init__(self):
        self.text = ""
        self.sentences = []
        # 专业术语和同义词映射
        self.term_map = {
            "深度学习": ["deep learning", "神经网络"],
            "传统机器学习": ["传统学习", "传统方法", "机器学习"],
            "量子计算": ["量子计算机"],
            "GPT-3": ["gpt3", "gpt 3"]
        }
        # 问题类型关键词
        self.question_types = {
            "区别": ["区别", "不同", "差异", "比较"],
            "事件": ["事件", "发生", "成果", "突破"],
            "问题": ["问题", "缺点", "局限", "挑战", "争议"]
        }

    def load_text(self, text):
        """加载并预处理文本"""
        self.text = text
        # 改进的句子分割，保留专业术语
        sentences = re.split(r'(?<=[。！？；\.!?;\n])\s*', text)
        self.sentences = [s.strip() for s in sentences if s.strip()]

    def answer_question(self, question):
        """核心问答逻辑"""
        # 1. 问题解析
        q_type, keywords = self._parse_question(question)

        # 2. 句子检索
        scored_sentences = []
        for sent in self.sentences:
            score = self._calculate_score(sent, keywords, q_type)
            if score > 0:
                scored_sentences.append((score, sent))

        # 3. 答案生成
        return self._generate_answer(q_type, scored_sentences)

    def _parse_question(self, question):
        """解析问题类型和关键词"""
        # 识别问题类型
        q_type = None
        for typ, words in self.question_types.items():
            if any(w in question for w in words):
                q_type = typ
                break

        # 提取关键词
        keywords = set()
        # 添加显式关键词
        keywords.update(re.findall(r'[\w\-]+', question.lower()))
        # 添加术语映射
        for term, aliases in self.term_map.items():
            if term in question:
                keywords.update(aliases)
                keywords.add(term.lower())
        # 移除停用词
        stopwords = {"什么", "怎么", "如何", "哪些", "的", "是", "在"}
        keywords = {w for w in keywords if w not in stopwords and len(w) > 1}

        return q_type, keywords

    def _calculate_score(self, sentence, keywords, q_type):
        """计算句子相关性得分"""
        score = 0
        sent_lower = sentence.lower()

        # 关键词匹配
        for kw in keywords:
            if kw in sent_lower:
                score += 2 if len(kw) > 2 else 1  # 长关键词权重更高

        # 问题类型特征匹配
        if q_type in self.question_types:
            for word in self.question_types[q_type]:
                if word in sentence:
                    score += 1
                    break

        # 时间关键词特殊处理
        if q_type == "事件" and re.search(r'20\d{2}', sentence):
            score += 2

        return score

    def _generate_answer(self, q_type, scored_sentences):
        """生成最终答案"""
        if not scored_sentences:
            return "在文本中找不到相关答案"

        scored_sentences.sort(reverse=True, key=lambda x: x[0])

        # 区别类问题需要对比回答
        if q_type == "区别" and len(scored_sentences) >= 2:
            return (f"主要区别在于：\n1. {scored_sentences[0][1]}\n"
                    f"2. {scored_sentences[1][1]}")

        # 问题类需要负面表述
        elif q_type == "问题":
            problems = [s for _, s in scored_sentences
                        if any(w in s for w in ["但", "然而", "问题", "挑战"])]
            if problems:
                return "存在的问题：" + "；".join(problems[:2])

        # 默认返回最佳匹配
        return scored_sentences[0][1]


# 测试用例
if __name__ == "__main__":
    qa = OptimizedChineseQA()

    complex_text = """
    人工智能(AI)是模拟人类智能的计算机系统，可分为弱人工智能和强人工智能。弱人工智能专注于特定任务，如语音识别或图像分类。
    深度学习是机器学习的一个分支，使用多层神经网络处理复杂模式识别。2012年，AlexNet在ImageNet竞赛中取得突破性成果，推动了计算机视觉的发展。

    自然语言处理(NLP)使计算机能理解、解释和生成人类语言。Transformer架构(如BERT、GPT)的出现显著提升了NLP性能。
    2020年发布的GPT-3拥有1750亿参数，能生成高质量文本，但也存在偏见和错误信息问题。

    量子计算利用量子力学原理进行计算。与传统计算机不同，量子比特可以同时处于0和1的叠加态。
    Google在2019年宣称实现量子霸权，但IBM对此提出质疑。量子计算有望在密码学、材料科学领域带来革命。
    """

    qa.load_text(complex_text)

    questions = [
        "深度学习和传统机器学习的主要区别是什么？",
        "2012年在计算机视觉领域发生了什么重要事件？",
        "量子计算与传统计算在基本原理上有何不同？",
        "关于量子霸权存在什么争议？",
        "GPT-3存在哪些问题？"
    ]

    for q in questions:
        print(f"\n问题: {q}")
        print("答案:", qa.answer_question(q))
        print("-" * 60)