import networkx as nx
from collections import defaultdict
import pandas as pd
import ollama
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import pickle
import os
from functools import lru_cache
import time
import math
import re


class TourismRecommender:
    def __init__(self, kg_file='tourist_kg.gexf', attractions_csv='enhanced_attractions.csv'):
        """初始化推荐系统，自动启用缓存优化"""
        start_time = time.time()

        # 1. 知识图谱和景点数据缓存
        cache_file = 'kg_cache.pkl'
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.kg, self.attractions_dict = pickle.load(f)
            # 保持兼容性：从字典恢复DataFrame
            self.attractions_df = pd.DataFrame.from_dict(self.attractions_dict, orient='index')
            print(f"缓存加载成功，耗时 {time.time() - start_time:.2f}s")
        else:
            print("未找到缓存，重新初始化数据...")
            self.kg = nx.read_gexf(kg_file)
            self.attractions_df = pd.read_csv(attractions_csv, encoding='utf-8-sig')
            # 将DataFrame转为嵌套字典 {景点名称: {列名: 值}}
            self.attractions_dict = self.attractions_df.set_index('名称').to_dict('index')
            # 保存缓存
            with open(cache_file, 'wb') as f:
                pickle.dump((self.kg, self.attractions_dict), f)
            print(f"数据初始化完成，耗时 {time.time() - start_time:.2f}s")

        # 2. 预处理数据
        self._preprocess_data()

    def _preprocess_data(self):
        """预处理区域关系和属性集合"""
        self.themes = set()
        self.audiences = set()
        self.areas = set()

        # 区域邻接关系（徐州实际地理）
        self.adjacent_areas = {
            '云龙区': ['泉山区', '鼓楼区'],
            '泉山区': ['云龙区', '铜山区'],
            '鼓楼区': ['云龙区', '贾汪区'],
            '铜山区': ['泉山区', '睢宁县'],
            '贾汪区': ['鼓楼区', '邳州市'],
            '邳州市': ['贾汪区', '新沂市'],
            '新沂市': ['邳州市'],
            '睢宁县': ['铜山区'],
            '沛县': ['丰县'],
            '丰县': ['沛县']
        }

        # 区域间通勤时间（分钟）
        self.travel_times = {
            ('云龙区', '泉山区'): 20,
            ('云龙区', '鼓楼区'): 15,
            ('泉山区', '铜山区'): 25,
            ('鼓楼区', '贾汪区'): 40,
            ('铜山区', '睢宁县'): 60,
            ('贾汪区', '邳州市'): 50,
            ('邳州市', '新沂市'): 45,
            ('沛县', '丰县'): 30
        }
        # 使时间字典双向
        for (a, b), time in list(self.travel_times.items()):
            self.travel_times[(b, a)] = time

        # 提取所有主题、人群和区域
        for node in self.kg.nodes(data=True):
            node_type = node[1].get('type')
            name = node[0].split('_')[-1]
            if node_type == 'Theme':
                self.themes.add(name)
            elif node_type == 'Audience':
                self.audiences.add(name)
            elif node_type == 'Area':
                self.areas.add(name)

        # 节假日特殊规则
        self.holiday_special_hours = {
            '春节': {'开放变化': '延长2小时'},
            '国庆节': {'开放变化': '分时段预约'}
        }

    @lru_cache(maxsize=1000)
    def parse_opening_hours(self, opening_hours_str, visit_date=None):
        """带缓存的开放时间解析"""
        if pd.isna(opening_hours_str) or not opening_hours_str:
            return True

        opening_hours_str = str(opening_hours_str)
        if "全天开放" in opening_hours_str:
            return True
        if "不开放" in opening_hours_str or "闭馆" in opening_hours_str:
            if visit_date:
                date_str = visit_date.strftime('%m/%d')
                if date_str in opening_hours_str:
                    return False
            return "不开放" not in opening_hours_str

        if visit_date:
            date_str = visit_date.strftime('%m/%d')
            weekday = visit_date.weekday()
            is_weekend = weekday >= 5
            is_holiday = self._is_holiday(visit_date)

            rules = [r.strip() for r in opening_hours_str.split(';') if r.strip()]
            for rule in rules:
                if '-' in rule.split()[0]:
                    date_part, time_part = rule.split(maxsplit=1)
                    start_date, end_date = date_part.split('-')
                    if self._date_in_range(visit_date, start_date, end_date):
                        if "周一" in time_part and weekday != 0:
                            continue
                        if "周二" in time_part and weekday != 1:
                            continue
                        if "周三" in time_part and weekday != 2:
                            continue
                        if "周四" in time_part and weekday != 3:
                            continue
                        if "周五" in time_part and weekday != 4:
                            continue
                        if "周六" in time_part and weekday != 5:
                            continue
                        if "周日" in time_part and weekday != 6:
                            continue

                        # 检查节假日特殊规则
                        if any(h in time_part for h in
                               ["元旦节", "春节", "清明节", "劳动节", "端午节", "中秋节", "国庆节"]):
                            if not is_holiday:
                                continue

                        return True

            # 处理特定日期规则
            if date_str in opening_hours_str:
                return True
        return True

    def get_adjacent_areas(self, area):
        """获取相邻区域"""
        return self.adjacent_areas.get(area, [])

    def get_travel_time(self, area1, area2):
        """获取两个区域间的通勤时间"""
        if area1 == area2:
            return 0
        return self.travel_times.get((area1, area2), 90)  # 默认90分钟

    def _date_in_range(self, date, start_str, end_str):
        """检查日期是否在范围内"""
        try:
            start = self._parse_partial_date(start_str, date.year)
            end = self._parse_partial_date(end_str, date.year)
            return start <= date <= end
        except:
            return False

    def _parse_partial_date(self, date_str, year):
        """解析部分日期(MM/DD)"""
        month, day = map(int, date_str.split('/'))
        return datetime(year, month, day).date()

    def _is_holiday(self, date):
        """简单判断是否是节假日"""
        holidays = {
            (1, 1): "元旦节",
            (5, 1): "劳动节",
            (10, 1): "国庆节",
        }
        return (date.month, date.day) in holidays

    def check_opening_status(self, attraction_name, visit_date=None):
        """检查景点在指定日期是否开放"""
        try:
            record = self.attractions_df[self.attractions_df['名称'] == attraction_name].iloc[0]
            opening_info = record['opening_hours']
            return self.parse_opening_hours(opening_info, visit_date)
        except:
            return True  # 如果出错，默认开放

    def get_attractions(self):
        """获取所有景点节点"""
        return [n for n, data in self.kg.nodes(data=True) if data.get('type') == 'Attraction']

    def get_attraction_info(self, node):
        """优化后的景点信息获取（字典加速）"""
        info = self.kg.nodes[node]
        name = info.get('name', node.replace('Attraction_', ''))

        # 优先从字典获取CSV中的增强数据
        csv_data = self.attractions_dict.get(name, {})

        # 处理评论数（转换为数值）
        comment_count = 0
        if '评论数' in csv_data and isinstance(csv_data['评论数'], str):
            try:
                comment_count = int(re.sub(r'[^\d]', '', csv_data['评论数']))
            except:
                comment_count = 0

        # 处理评分可能为None的情况，并确保评分在合理范围内
        rating = csv_data.get('评分')
        if rating in ['无评分', '暂无评分', None]:
            # 基于图谱特征的评分预测
            themes_attr = self.get_related_attributes(node, 'HAS_THEME')
            audiences_attr = self.get_related_attributes(node, 'SUITABLE_FOR')
            theme_count = len(themes_attr)
            audience_count = len(audiences_attr)
            popularity = self.kg.degree(node)

            # 复合计算公式（百分制）
            rating = 70 + \
                     10 * math.log1p(theme_count) + \
                     8 * math.log1p(audience_count) + \
                     5 * math.log1p(popularity)
            rating = min(95, max(60, rating))  # 控制在60-95之间
            rating = rating / 20  # 转换为5分制
        else:
            try:
                rating = float(rating)
                rating = min(5.0, max(1.0, rating))  # 确保评分在1-5之间
            except (ValueError, TypeError):
                rating = 3.5  # 默认值

        return {
            'name': name,
            'rating': rating,
            'comment_count': comment_count,
            'address': info.get('address', csv_data.get('enhanced_address', '')),
            'opening_hours': csv_data.get('opening_hours', info.get('opening_hours', ''))
        }

    def recommend(self, themes=[], audiences=[], target_area=None, date_range=None, top_n=5):
        """改进版推荐算法，采用标准化评分转换"""
        candidates = []
        target_areas = [target_area] if target_area else []

        # 预计算最大评论数（避免重复计算）
        max_comments = max(1, max(
            a.get('评论数', 0) if isinstance(a.get('评论数'), (int, float)) else 0
            for a in self.attractions_dict.values()
        ))

        # 定义评论数区间和对应加分（更严格的区间划分）
        comment_intervals = [
            (0, 50, 1),  # 0-50评论：加1分
            (50, 200, 2),  # 50-200评论：加2分
            (200, 1000, 4),  # 200-1000评论：加4分
            (1000, 5000, 6),  # 1000-5000评论：加6分
            (5000, float('inf'), 8)  # 5000+评论：加8分
        ]

        # 理论最高分计算（调整后）
        MAX_RAW_SCORE = 100  # 调整为100分制：30+25+20+10+8+5 = 98≈100

        for attraction in self.get_attractions():
            info = self.get_attraction_info(attraction)
            themes_attr = self.get_related_attributes(attraction, 'HAS_THEME')
            audiences_attr = self.get_related_attributes(attraction, 'SUITABLE_FOR')

            # 1. 计算主题匹配度（权重从40降到30分）
            theme_score = 30 * (1.0 if not themes else len(set(themes) & set(themes_attr)) / len(themes))

            # 2. 计算人群匹配度（权重从30降到25分）
            audience_score = 25 * (1.0 if not audiences else len(set(audiences) & set(audiences_attr)) / len(audiences))

            # 3. 计算区域匹配度（保持20分）
            area = self.extract_area_from_address(info['address'])
            if not target_areas:
                area_score = 20
            elif area in target_areas:
                area_score = 20
            elif area in self.get_adjacent_areas(target_areas[0]):
                area_score = 14  # 20*0.7
            else:
                area_score = 8  # 20*0.4

            # 4. 基础评分（保持10分）
            base_rating_score = 10 * (info['rating'] / 5.0)

            # 5. 开放时间影响（扣分制，最多扣5分）
            open_penalty = 0
            if date_range:
                open_days = sum(self.check_opening_status(info['name'], date) for date in date_range)
                open_ratio = open_days / len(date_range) if open_days > 0 else 0
                if open_ratio <= 0:  # 完全不开放
                    continue
                elif open_ratio < 0.8:  # 开放时间不足
                    open_penalty = 5 - 5 * open_ratio

            # 6. 评论数热度（区间制加分，上限从10降到8分）
            comment_bonus = 0
            for min_c, max_c, bonus in comment_intervals:
                if min_c <= info['comment_count'] < max_c:
                    comment_bonus = bonus
                    break

            # 7. 季节性调整（上限从5降到3分）
            seasonal_bonus = 0
            if date_range and len(date_range) > 0:
                month = date_range[0].month
                if '冰雪' in themes_attr and month in [12, 1, 2]:
                    seasonal_bonus = 2  # 从3降到2
                elif '赏花' in themes_attr and month in [3, 4]:
                    seasonal_bonus = 1  # 从2降到1

            # 计算原始总分（百分制）
            raw_score = (
                    theme_score +
                    audience_score +
                    area_score +
                    base_rating_score +
                    comment_bonus +
                    seasonal_bonus -
                    open_penalty
            )

            # 标准化分数转换（关键修正）
            # 线性转换（推荐）
            final_score = round(1 + 4 * (raw_score / MAX_RAW_SCORE), 1)

            # 确保分数在1-5范围内
            final_score = min(5.0, max(1.0, final_score))

            candidates.append((attraction, final_score, info))

        # 按分数+评论数双重排序
        candidates.sort(key=lambda x: (-x[1], -x[2]['comment_count']))

        return candidates[:top_n]

    def generate_travel_plan(self, selected_attractions, date_range=None):
        """生成完整旅游计划"""
        attractions_info = []
        for attr in selected_attractions:
            info = self.get_attraction_info(attr[0])
            themes = self.get_related_attributes(attr[0], 'HAS_THEME')
            description = self.generate_attraction_description(info['name'])

            # 获取更精确的开放时间
            try:
                csv_record = self.attractions_df[self.attractions_df['名称'] == info['name']].iloc[0]
                info['opening_hours'] = csv_record['opening_hours']
            except:
                pass

            attractions_info.append({
                'name': info['name'],
                'address': info['address'],
                'opening_hours': info['opening_hours'],
                'themes': themes,
                'description': description,
                'score': attr[1],
                'area': self.extract_area_from_address(info['address']),
                'comment_count': info['comment_count']
            })

        # 优化路径规划：按区域聚类并计算最优路线
        if len(attractions_info) > 1:
            attractions_info = self._optimize_route(attractions_info)

        # 如果有日期区间，按天分组景点
        if date_range:
            # 将景点分配到各天
            daily_attractions = [[] for _ in date_range]
            for i, attr in enumerate(attractions_info):
                day_idx = i % len(date_range)
                daily_attractions[day_idx].append(attr)

            # 为每天生成计划
            date_plans = []
            for day, day_attrs in zip(date_range, daily_attractions):
                prompt = self._build_llm_prompt(day_attrs, day.strftime('%Y-%m-%d'))
                try:
                    response = ollama.generate(
                        model='llama3.2',
                        prompt=prompt,
                        options={'temperature': 0.7}
                    )
                    date_plans.append({
                        'date': day.strftime('%Y年%m月%d日'),
                        'plan': response['response']
                    })
                except Exception as e:
                    print(f"LLM生成失败: {str(e)}")
                    date_plans.append({
                        'date': day.strftime('%Y年%m月%d日'),
                        'plan': f"无法生成{day.strftime('%Y年%m月%d日')}的智能规划，请参考景点信息自行安排"
                    })

            # 添加天气提示
            weather_suggestion = self._get_weather_tips(date_range[0]) if date_range else None

            # 添加美食推荐
            food_recommendation = self._generate_food_recommendation()

            return {
                'summary': f"{len(selected_attractions)}个精选景点 · 预计总游览时间{self._estimate_total_time(attractions_info)}小时",
                'attractions': attractions_info,
                'date_plans': date_plans,
                'extras': {
                    'food': food_recommendation,
                    'weather': weather_suggestion
                }
            }
        else:
            # 单日计划
            prompt = self._build_llm_prompt(attractions_info, None)
            try:
                response = ollama.generate(
                    model='llama3.2',
                    prompt=prompt,
                    options={'temperature': 0.7}
                )

                # 添加美食推荐
                food_recommendation = self._generate_food_recommendation()

                return {
                    'summary': f"{len(selected_attractions)}个精选景点 · 预计游览时间{self._estimate_total_time(attractions_info)}小时",
                    'attractions': attractions_info,
                    'plan': response['response'],
                    'extras': {
                        'food': food_recommendation
                    }
                }
            except Exception as e:
                print(f"LLM生成失败: {str(e)}")
                return {
                    'summary': f"{len(selected_attractions)}个精选景点",
                    'attractions': attractions_info,
                    'plan': "无法生成智能规划，请参考景点信息自行安排",
                    'extras': {
                        'food': self._generate_food_recommendation()
                    }
                }

    def _optimize_route(self, attractions):
        """改进的路线优化算法"""
        if len(attractions) <= 2:
            return attractions

        # 构建完整的通勤时间矩阵
        locations = [(attr['area'], attr['name']) for attr in attractions]
        time_matrix = [
            [self._get_actual_travel_time(start[0], end[0])
             for end in locations]
            for start in locations
        ]

        # 使用贪心算法+2-opt优化
        optimized_indices = self._solve_tsp(time_matrix)
        return [attractions[i] for i in optimized_indices]

    def _get_actual_travel_time(self, area1, area2):
        """考虑多种交通方式的综合时间"""
        base_time = self.get_travel_time(area1, area2)
        # 同区域景点间增加15分钟缓冲
        if area1 == area2:
            return 15
            # 跨区域景点增加30分钟停车/找路时间
        return base_time + 30

    def _solve_tsp(self, time_matrix):
        """解决旅行商问题的简化实现"""
        n = len(time_matrix)
        if n <= 1:
            return [0]

        # 最近邻算法
        path = [0]
        unvisited = set(range(1, n))

        while unvisited:
            last = path[-1]
            next_node = min(unvisited, key=lambda x: time_matrix[last][x])
            path.append(next_node)
            unvisited.remove(next_node)

        # 2-opt优化
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1: continue
                    # 检查交换是否缩短距离
                    delta = (time_matrix[path[i - 1]][path[j]] +
                             time_matrix[path[i]][path[j + 1] if j + 1 < n else 0] -
                             time_matrix[path[i - 1]][path[i]] -
                             time_matrix[path[j]][path[j + 1] if j + 1 < n else 0])
                    if delta < 0:
                        path[i:j + 1] = path[j:i - 1:-1]  # 反转子路径
                        improved = True

        return path

    def _estimate_total_time(self, attractions):
        """估算总游览时间"""
        base_time = len(attractions) * 2  # 每个景点平均2小时
        travel_time = 0

        # 计算景点间通勤时间
        for i in range(len(attractions) - 1):
            area1 = attractions[i]['area']
            area2 = attractions[i + 1]['area']
            travel_time += self._get_actual_travel_time(area1, area2) / 60  # 转换为小时

        return round(base_time + travel_time, 1)

    def _get_weather_tips(self, date):
        """简单的天气提示（实际应用中可接入天气API）"""
        month = date.month
        if month in [12, 1, 2]:
            return "冬季出行建议：徐州冬季较冷，建议穿着羽绒服等保暖衣物"
        elif month in [3, 4, 5]:
            return "春季出行建议：徐州春季气候宜人，建议携带薄外套"
        elif month in [6, 7, 8]:
            return "夏季出行建议：徐州夏季炎热，建议做好防晒措施"
        else:
            return "秋季出行建议：徐州秋季凉爽，适合户外活动"

    def _generate_food_recommendation(self):
        """生成徐州特色美食推荐"""
        xuzhou_foods = [
            {"name": "徐州地锅鸡", "description": "徐州传统名菜，鸡肉鲜嫩，锅贴饼吸满汤汁"},
            {"name": "沛县狗肉", "description": "历史悠久的传统美食，肉质鲜美"},
            {"name": "徐州羊肉汤", "description": "冬季暖身佳品，汤白味鲜"},
            {"name": "烙馍卷馓子", "description": "徐州特色小吃，香脆可口"}
        ]

        restaurants = [
            {"name": "徐州印象", "address": "云龙区和平路58号", "specialty": "地锅鸡、羊肉汤"},
            {"name": "老地方菜馆", "address": "泉山区解放南路12号", "specialty": "沛县狗肉、家常菜"},
            {"name": "彭城风味", "address": "鼓楼区中山北路32号", "specialty": "徐州传统小吃"}
        ]

        return {
            "foods": xuzhou_foods,
            "restaurants": restaurants
        }

    def _build_llm_prompt(self, attractions_info, visit_date):
        """构建LLM提示词"""
        # 计算景点间的通勤时间
        travel_times = []
        for i in range(len(attractions_info) - 1):
            current_area = attractions_info[i]['area']
            next_area = attractions_info[i + 1]['area']
            time = self.get_travel_time(current_area, next_area)
            travel_times.append(f"{attractions_info[i]['name']}到{attractions_info[i + 1]['name']}大约需要{time}分钟")

        prompt = """你是一个专业的徐州旅游规划师。请根据以下景点信息生成一份详细的中文旅游计划：

    旅行日期: {date}
    景点列表: {attraction_names}

    计划要求:
    1. 按合理路线顺序排列景点，考虑区域间的通勤时间
    2. 包含每个景点的特色介绍(50字左右)
    3. 提供景点间的交通建议和预计时间
    4. 给出合理的休息建议
    5. 如果提供了具体日期，确保考虑开放时间
    6. 不要包含餐饮建议，餐饮会单独推荐
    7. 所有内容请使用中文

    通勤时间参考:
    {travel_times}

    景点详细信息:
    {details}

    请按照以下格式生成计划:
    【上午安排】
    景点1名称 (预计游览时间)
    - 特色介绍
    - 交通建议

    【下午安排】
    景点2名称 (预计游览时间)
    - 特色介绍
    - 交通建议
    """.format(
            date=visit_date if visit_date else "未指定日期",
            attraction_names="、".join([a['name'] for a in attractions_info]),
            travel_times="\n".join(travel_times),
            details="\n".join([
                f"{i + 1}. {attr['name']} (评分:{attr['score']:.1f}/5, 评论数:{attr['comment_count']})\n"
                f"   地址: {attr['address']}\n"
                f"   开放时间: {attr['opening_hours']}\n"
                f"   特色: {'、'.join(attr['themes'])}\n"
                f"   简介: {attr['description']}\n"
                f"   所在区域: {attr['area']}"
                for i, attr in enumerate(attractions_info)
            ])
        )
        return prompt

    def extract_area_from_address(self, address):
        """从地址中提取区域"""
        for area in self.areas:
            if area in address:
                return area
        return "未知区域"

    def get_related_attributes(self, node, relation):
        """获取关联属性"""
        return [n.split('_')[-1] for _, n, data in self.kg.edges(node, data=True)
                if data.get('relation') == relation]

    def generate_attraction_description(self, name):
        """增强版景点描述生成"""
        # 优先从CSV获取预定义的描述
        csv_desc = self.attractions_dict.get(name, {}).get('description')
        if csv_desc and not pd.isna(csv_desc):
            return csv_desc

        # 动态生成时提供更详细提示
        themes = self.get_related_attributes(f'Attraction_{name}', 'HAS_THEME')
        prompt = f"""作为徐州旅游专家，请用60字生动描述{name}（特色：{'、'.join(themes)}）：
            - 开头用1个四字成语概括
            - 突出2-3个最独特亮点
            - 结尾用"推荐...人群体验"句式
            示例：气势恢宏的汉文化遗址，栩栩如生的汉代石刻艺术与..."""

        try:
            response = ollama.generate(
                model='llama3.2',
                prompt=prompt,
                options={'temperature': 0.3}  # 降低随机性
            )
            return self._postprocess_description(response['response'])
        except:
            return f"{name}是徐州{themes[0] if themes else '著名'}景点，值得体验"

    def _postprocess_description(self, desc):
        """后处理生成的描述文本"""
        # 移除多余的换行和空格
        desc = ' '.join(desc.split())
        # 确保以句号结尾
        if not desc.endswith(('。', '!', '?')):
            desc += '。'
        return desc


if __name__ == '__main__':
    # 测试代码
    recommender = TourismRecommender()
    print("=== 测试推荐 ===")
    recs = recommender.recommend(themes=["两汉文化"], audiences=["文化爱好者"])
    for i, (attr, score, info) in enumerate(recs[:5]):  # 展示前5个结果
        print(f"{i + 1}. {info['name']} (评分: {score:.1f}/5, 评论数: {info['comment_count']})")

    print("\n=== 测试旅游计划生成 ===")
    plan = recommender.generate_travel_plan(recs[:3])  # 为前3个景点生成计划
    print(plan['plan'])
    print("\n=== 美食推荐 ===")
    print(plan['extras']['food'])