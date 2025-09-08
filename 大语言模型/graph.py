import pandas as pd
import networkx as nx
import re
import ast
from collections import defaultdict
import ollama
import time
from tqdm import tqdm  # 进度条显示

# 徐州区县列表（用于验证模型输出）
XUZHOU_DISTRICTS = {
    '云龙区', '鼓楼区', '泉山区', '铜山区', '贾汪区',
    '邳州市', '新沂市', '睢宁县', '沛县', '丰县'
}


# --- 地址增强模块 ---
class AddressEnhancer:
    def __init__(self):
        self.district_cache = {}  # 缓存已知的地址-区县映射
        self.known_mapping = {  # 预定义的地址映射
            "汉城南路1号": "沛县",
            "宝莲寺路5号": "鼓楼区",
            "襄王北路3号": "鼓楼区",
            "湖东路": "泉山区",
            "兵马俑路1号": "云龙区",
            "解放南路2号": "泉山区",
            "和平路118号": "云龙区",
            "湖中路": "泉山区",
            "玉带大道9号": "泉山区",
            "九里山东路79号": "鼓楼区"
        }

    def extract_district(self, full_address):
        """从完整地址中提取区县"""
        if not isinstance(full_address, str):
            return None

        match = re.search(r'徐州市(.+?[区县市])', full_address)
        return match.group(1) if match else None

    def query_llama(self, short_address):
        """使用本地LLaMA模型查询区县信息"""
        prompt = f"""
        你是一个熟悉江苏省徐州市地理的专家。请根据以下地址判断它位于徐州市的哪个区县。
        只回答区县名称，不要包含其他任何文字。

        已知徐州市的区县包括：云龙区、鼓楼区、泉山区、铜山区、贾汪区、邳州市、新沂市、睢宁县、沛县、丰县。

        地址示例：
        "汉城南路1号" -> "沛县"
        "宝莲寺路5号" -> "鼓楼区"
        "湖东路" -> "泉山区"

        现在请判断以下地址：
        地址：{short_address}
        """

        try:
            response = ollama.generate(
                model='llama3.2',
                prompt=prompt,
                options={'temperature': 0.0}  # 完全确定性输出
            )
            district = response['response'].strip()
            # 清理模型输出
            district = re.sub(r'[^云龙区鼓楼区泉山区铜山区贾汪区邳州市新沂市睢宁县沛县丰县]', '', district)
            return district if district in XUZHOU_DISTRICTS else None
        except Exception as e:
            print(f"模型查询失败: {str(e)}")
            return None

    def enhance_address(self, original_address):
        """增强地址信息"""
        if not isinstance(original_address, str):
            return original_address

        # 如果已有完整区县信息，直接使用
        district = self.extract_district(original_address)
        if district:
            return original_address

        # 检查预定义映射
        for short_addr, dist in self.known_mapping.items():
            if short_addr in original_address:
                return f"江苏省徐州市{dist}{original_address}"

        # 检查缓存
        if original_address in self.district_cache:
            district = self.district_cache[original_address]
            return f"江苏省徐州市{district}{original_address}"

        # 使用模型查询
        district = self.query_llama(original_address)
        if district:
            self.district_cache[original_address] = district
            return f"江苏省徐州市{district}{original_address}"

        # 无法确定的情况
        print(f"无法确定地址所属区县: {original_address}")
        return original_address


# --- 知识图谱构建模块 ---
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_counter = defaultdict(int)

    def add_node(self, node_type, name, **attrs):
        node_id = f"{node_type}_{name}"
        if node_id not in self.graph:
            attrs['type'] = node_type
            self.graph.add_node(node_id, **attrs)
            self.node_counter[node_type] += 1
        return node_id

    def add_edge(self, source_id, target_id, relation_type, weight=1):
        self.graph.add_edge(source_id, target_id, relation=relation_type, weight=weight)

    def build_from_dataframe(self, df):
        print("正在构建知识图谱...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # 添加景点节点
            attr_node = self.add_node(
                'Attraction',
                row['名称'],
                rating=row['评分'],
                review_count=row['评论数'],
                address=row['enhanced_address'],
                opening_hours=row['opening_hours']
            )

            # 添加主题关系（带权重）
            for theme, weight in row['主题详情'].items():
                theme_node = self.add_node('Theme', theme)
                self.add_edge(attr_node, theme_node, 'HAS_THEME', weight=weight)

            # 添加人群关系（带权重）
            for group, weight in row['人群详情'].items():
                group_node = self.add_node('Audience', group)
                self.add_edge(attr_node, group_node, 'SUITABLE_FOR', weight=weight)

            # 添加区域关系
            district = self.extract_district(row['enhanced_address'])
            if district:
                area_node = self.add_node('Area', district)
                self.add_edge(attr_node, area_node, 'LOCATED_IN')

        print(f"图谱构建完成，共 {len(self.graph.nodes)} 个节点，{len(self.graph.edges)} 条边")

    def extract_district(self, address):
        """从增强后的地址提取区县"""
        match = re.search(r'徐州市(.+?[区县市])', address)
        return match.group(1) if match else None


# --- 可视化模块 ---
def visualize_graph(kg, output_file='knowledge_graph.html'):
    try:
        from pyvis.network import Network
        net = Network(
            height="800px",
            width="100%",
            directed=True,
            notebook=False,
            cdn_resources='in_line'
        )

        # 节点颜色和大小配置
        color_map = {
            'Attraction': '#FFA07A',
            'Theme': '#98FB98',
            'Audience': '#ADD8E6',
            'Area': '#DDA0DD'
        }
        size_map = {
            'Attraction': 20,
            'Theme': 15,
            'Audience': 15,
            'Area': 15
        }

        # 添加节点
        for node in kg.graph.nodes:
            node_type = kg.graph.nodes[node]['type']
            label = node.split('_')[1] if '_' in node else node
            title = f"<b>{label}</b><br>类型: {node_type}"

            if node_type == 'Attraction':
                attrs = kg.graph.nodes[node]
                title += f"<br>评分: {attrs.get('rating', '无')}"
                title += f"<br>地址: {attrs.get('address', '无')}"

            net.add_node(
                node,
                label=label,
                color=color_map[node_type],
                title=title,
                size=size_map[node_type]
            )

        # 添加边（带权重）
        for u, v, data in kg.graph.edges(data=True):
            net.add_edge(
                u, v,
                title=f"{data['relation']} (权重:{data['weight']})",
                width=data['weight'] * 0.5
            )

        # 物理布局配置
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            }
        }
        """)

        net.show(output_file)
        print(f"可视化已保存到 {output_file}")

    except Exception as e:
        print(f"可视化失败: {str(e)}")
        print("请确保已安装pyvis: pip install pyvis")


# --- 主程序 ---
def main():
    # 1. 数据加载与预处理
    print("正在加载数据...")
    df = pd.read_csv('classified_attractions_improved.csv', encoding='utf-8-sig')

    # 处理主题和人群
    df['主题列表'] = df['主题'].str.split(',').apply(lambda x: [t.strip() for t in x] if isinstance(x, str) else [])
    df['人群列表'] = df['适合人群'].str.split(',').apply(lambda x: [g.strip() for g in x] if isinstance(x, str) else [])

    # 处理详情字段
    df['主题详情'] = df['主题详情'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x.startswith('{') else {})
    df['人群详情'] = df['人群详情'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x.startswith('{') else {})

    # 2. 地址增强
    print("正在增强地址信息...")
    enhancer = AddressEnhancer()
    tqdm.pandas(desc="地址补全进度")
    df['enhanced_address'] = df['信息'].progress_apply(enhancer.enhance_address)

    # 保存增强后的数据
    df.to_csv('enhanced_attractions.csv', index=False, encoding='utf-8-sig')
    print("已保存增强后的数据到 enhanced_attractions.csv")

    # 3. 构建知识图谱
    kg = KnowledgeGraph()
    kg.build_from_dataframe(df)

    # 4. 分析与可视化
    print("\n知识图谱统计信息:")
    print(f"- 景点数量: {kg.node_counter['Attraction']}")
    print(f"- 主题类型: {kg.node_counter['Theme']}")
    print(f"- 人群分类: {kg.node_counter['Audience']}")
    print(f"- 区域数量: {kg.node_counter['Area']}")

    # 计算重要景点（基于加权连接数）
    weighted_degree = dict(kg.graph.degree(weight='weight'))
    top_attractions = sorted(
        [(n.split('_')[1], d) for n, d in weighted_degree.items() if 'Attraction' in n],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    print("\n重要景点Top5(基于连接权重):")
    for name, score in top_attractions:
        print(f"  {name} (连接权重: {score})")

    # 可视化
    visualize_graph(kg)


if __name__ == "__main__":
    main()