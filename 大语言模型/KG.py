import json
import stanza
import ollama
import networkx as nx
from pyvis.network import Network
from typing import List, Dict, Optional


class KnowledgeGraphBuilder:
    def __init__(self, llm_model: str = "llama3"):
        """
        初始化知识图谱构建器（完全离线版）
        :param llm_model: 本地Ollama模型名称
        """
        # 中文NLP模型
        stanza.download('zh')
        self.nlp_stanza = stanza.Pipeline('zh')

        # 知识图谱数据
        self.triples = []
        self.graph = nx.DiGraph()
        self.llm_model = llm_model

    def extract_entities(self, text: str) -> List[Dict]:
        """使用Stanza抽取中文实体并增强类型识别"""
        doc = self.nlp_stanza(text)
        entities = []
        for sent in doc.sentences:
            for ent in sent.ents:
                # 增强实体类型识别
                ent_type = ent.type
                if "公司" in ent.text or "集团" in ent.text or "企业" in ent.text:
                    ent_type = "ORG"
                elif "年" in ent.text and any(c.isdigit() for c in ent.text):
                    ent_type = "DATE"
                elif "市" in ent.text or "省" in ent.text or "州" in ent.text:
                    ent_type = "LOC"

                entities.append({
                    "text": ent.text,
                    "type": ent_type,
                    "start_pos": ent.start_char,
                    "end_pos": ent.end_char
                })
        return entities

    def build_kg_from_text(self, text: str) -> None:
        """从文本构建知识图谱"""
        try:
            extracted_triples = self.extract_relations(text)
            if extracted_triples:
                self.triples.extend(extracted_triples)
                self._update_graph()
        except Exception as e:
            print(f"构建知识图谱时出错: {str(e)}")

    def _update_graph(self) -> None:
        """更新图结构"""
        for triple in self.triples:
            if not all(key in triple for key in ['head', 'relation', 'tail']):
                continue

            # 添加节点属性
            self.graph.add_node(triple['head'], label=triple['head'], title=triple['head'])
            self.graph.add_node(triple['tail'], label=triple['tail'], title=triple['tail'])

            # 添加边属性
            self.graph.add_edge(
                triple['head'],
                triple['tail'],
                label=triple['relation'],
                title=triple['relation'],
                relation=triple['relation']
            )

    def extract_relations(self, text: str) -> List[Dict]:
        """使用Ollama生成三元组（完全离线）"""
        prompt = f"""
        请从以下文本中精确提取关系三元组（格式：头实体|关系|尾实体），特别注意：
        1. 收购关系必须严格包含年份，格式为"收购方|YYYY年收购|被收购方"
        2. 创始人关系使用"创始人"标签
        3. CEO关系使用"CEO"标签
        4. 总部关系使用"总部位于"标签
        5. 产品关系使用"产品"标签
        6. 时间关系使用"成立于"标签

        示例正确格式：
        微软|2016年收购|LinkedIn
        Facebook|创始人|马克·扎克伯格
        苹果|CEO|蒂姆·库克
        华为|总部位于|深圳

        文本：{text}

        请严格按以下格式输出，不要解释：
        头实体|关系|尾实体
        ...
        """
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                stream=False
            )
            triples = []
            for line in response["response"].split("\n"):
                line = line.strip()
                if "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        # 标准化关系标签
                        relation = parts[1]
                        if "收购" in relation:
                            # 提取年份，格式化为"YYYY年收购"
                            year_digits = ''.join(filter(str.isdigit, relation))
                            if year_digits:
                                relation = f"{year_digits[:4]}年收购"
                            else:
                                continue  # 跳过没有明确年份的收购关系

                        triples.append({
                            "head": parts[0],
                            "relation": relation,
                            "tail": "|".join(parts[2:])
                        })
            return triples
        except Exception as e:
            print(f"提取关系时出错: {str(e)}")
            return []

    def query_kg(self, question: str, method: str = "llm") -> str:
        """
        改进的知识图谱查询方法
        """
        if method == "triple":
            # 精确匹配
            results = []
            for triple in self.triples:
                if all(word.lower() in f"{triple['head']} {triple['relation']} {triple['tail']}".lower()
                       for word in question.split()):
                    results.append(f"{triple['head']} → {triple['relation']} → {triple['tail']}")

            if results:
                return "\n".join(results)
            return "未找到相关信息"

        elif method == "llm":
            # 构建结构化上下文
            context = "知识库中的信息：\n"

            # 安全地处理收购事件排序
            acquisitions = []
            for t in self.triples:
                if "收购" in t["relation"]:
                    try:
                        year = int(t["relation"].split("年")[0])
                        acquisitions.append((year, t))
                    except (ValueError, IndexError):
                        continue

            if acquisitions:
                context += "\n收购事件（按年份排序）：\n"
                for year, acq in sorted(acquisitions, key=lambda x: x[0]):
                    context += f"- {acq['head']} 于 {acq['relation']} {acq['tail']}\n"

            # 添加其他关系
            other_relations = [t for t in self.triples if "收购" not in t["relation"]]
            if other_relations:
                context += "\n其他关系：\n"
                for rel in other_relations:
                    context += f"- {rel['head']} {rel['relation']} {rel['tail']}\n"

            prompt = f"""
            {context}

            请基于以上知识准确回答问题：
            1. 直接回答，不要解释
            2. 如果不知道，回答"根据现有知识无法确定"
            3. 对于人物关系问题，直接引用知识库中的关系

            问题：{question}
            答案：
            """

            try:
                response = ollama.generate(
                    model=self.llm_model,
                    prompt=prompt,
                    stream=False
                )
                return response["response"].strip()
            except Exception as e:
                return f"查询时出错: {str(e)}"

    def visualize_kg(self, output_path: str = "kg_visualization.html") -> None:
        """可视化知识图谱（带关系标签）"""
        try:
            net = Network(
                notebook=True,
                directed=True,
                height="750px",
                width="100%",
                bgcolor="#ffffff",
                font_color="#333333"
            )

            # 按节点度设置大小
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1

            # 添加节点
            for node in self.graph.nodes():
                size = 15 + 20 * (degrees.get(node, 1) / max_degree)
                net.add_node(
                    node,
                    label=node,
                    title=node,
                    size=size,
                    color="#D2E5FF" if "收购" in node or "收购" in str(self.graph.nodes[node]) else "#FFD2D2"
                )

            # 添加边
            for edge in self.graph.edges(data=True):
                net.add_edge(
                    edge[0], edge[1],
                    label=edge[2]['label'],
                    title=edge[2]['title'],
                    width=2 if "收购" in edge[2]['label'] else 1
                )

            # 优化可视化参数
            net.set_options("""
            {
                "nodes": {
                    "font": {"size": 16},
                    "shape": "box",
                    "borderWidth": 2,
                    "color": {"border": "#2B7CE9", "background": "#D2E5FF"}
                },
                "edges": {
                    "arrows": {"to": {"enabled": true}},
                    "font": {"size": 12, "align": "middle"},
                    "color": {"color": "#2B7CE9", "highlight": "#FF0000"},
                    "smooth": {"type": "continuous"}
                },
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -100,
                        "centralGravity": 0.01,
                        "springLength": 150
                    },
                    "minVelocity": 0.75,
                    "solver": "forceAtlas2Based"
                }
            }
            """)
            net.show(output_path)
        except Exception as e:
            print(f"可视化时出错: {str(e)}")

    def save_triples(self, file_path: str, format: str = "json") -> None:
        """保存三元组数据"""
        try:
            data = []
            for triple in self.triples:
                if isinstance(triple, dict):
                    data.append({
                        "subject": str(triple.get("head", "")),
                        "relation": str(triple.get("relation", "")),
                        "object": str(triple.get("tail", ""))
                    })

            if format == "json":
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif format == "csv":
                import csv
                with open(file_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["subject", "relation", "object"])
                    writer.writeheader()
                    writer.writerows(data)
        except Exception as e:
            print(f"保存三元组时出错: {str(e)}")


if __name__ == "__main__":
    # 初始化知识图谱构建器
    kg_builder = KnowledgeGraphBuilder(llm_model="llama3.2")

    # 示例中文文本
    text = """
    苹果公司(Apple Inc.)由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩于1976年4月1日创立。这家总部位于加利福尼亚州库比蒂诺的科技巨头，目前由蒂姆·库克担任CEO。苹果最著名的产品包括iPhone智能手机、Mac电脑和iPad平板电脑。

    微软(Microsoft)则由比尔·盖茨和保罗·艾伦在1975年创立，现任CEO是萨提亚·纳德拉。这家总部位于华盛顿州雷德蒙德的公司，以Windows操作系统和Office办公软件闻名于世。2016年，微软以262亿美元收购了职业社交平台LinkedIn。

    在中国，华为技术有限公司成立于1987年，创始人任正非将这家深圳企业打造成了全球领先的通信设备供应商。华为的5G技术在全球处于领先地位，其智能手机品牌也享誉世界。阿里巴巴集团由马云带领18位创始人在1999年创立，总部位于杭州，核心业务包括淘宝、天猫和阿里云。

    2018年，谷歌母公司Alphabet的CEO桑达尔·皮查伊宣布以50亿美元收购智能穿戴设备制造商Fitbit。而在此前的2014年，Facebook（现Meta）以190亿美元收购了即时通讯应用WhatsApp，这笔交易由马克·扎克伯格亲自推动。

    特斯拉汽车公司由埃隆·马斯克在2003年创立，这家电动汽车制造商的总部设在美国帕洛阿尔托。2020年，特斯拉上海超级工厂开始量产Model 3，这是特斯拉在美国以外的首个超级工厂。
    """

    # 构建知识图谱
    print("构建知识图谱中...")
    kg_builder.build_kg_from_text(text)

    # 保存三元组
    kg_builder.save_triples("knowledge_triples.json")
    kg_builder.save_triples("knowledge_triples.csv", format="csv")

    # 可视化知识图谱
    print("生成可视化文件...")
    kg_builder.visualize_kg()

    # 知识图谱问答
    print("\n知识图谱问答测试：")
    questions = [
        "苹果公司的CEO是谁？",
        "华为的总部在哪里？",
        "哪些公司被微软收购了",
        "马云创建了哪些企业？",
        "特斯拉创立是在哪一年？",
        "Facebook收购WhatsApp是在微软收购LinkedIn之前还是之后？",
        "苹果公司有哪些产品？"
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {kg_builder.query_kg(q)}")