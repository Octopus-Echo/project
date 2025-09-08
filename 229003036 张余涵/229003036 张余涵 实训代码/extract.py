import pandas as pd
import re
from collections import defaultdict

# 读取CSV文件
df = pd.read_csv('attractions_add.csv', encoding='utf-8-sig')

# 更细致的主题关键词映射（增加权重）
theme_keywords = {
    # 文化历史类
    "两汉文化": {
        "keywords": ["汉墓", "楚王", "兵马俑", "汉代", "西汉", "东汉", "汉画像", "汉文化", "刘邦", "项羽", "彭城",
                     "戏马台"], "weight": 3},
    "历史遗迹": {"keywords": ["遗址", "古迹", "陵墓", "古镇", "古建筑", "古城", "古街", "考古", "出土"], "weight": 3},
    "博物馆": {"keywords": ["博物馆", "纪念馆", "展览馆", "陈列馆", "藏品", "文物", "馆藏"], "weight": 3},
    "宗教文化": {"keywords": ["寺", "庙", "禅", "佛教", "道教", "禅寺", "宝莲", "行宫", "教堂"], "weight": 2},

    # 自然风光类
    "自然风光": {"keywords": ["湖", "山", "河", "公园", "风景区", "湿地", "森林", "风景", "景观", "生态"], "weight": 2},
    "城市公园": {"keywords": ["公园", "广场", "绿地", "花园", "休闲区"], "weight": 1},

    # 娱乐体验类
    "主题乐园": {"keywords": ["乐园", "方特", "欢乐世界", "游乐", "过山车", "熊出没", "娱乐"], "weight": 2},
    "水上活动": {"keywords": ["水上", "游泳", "冲浪", "水世界", "海浪", "浴场", "游船", "漂流"], "weight": 2},
    "冬季运动": {"keywords": ["滑雪", "滑冰", "雪场", "冰雪"], "weight": 2},
    "文化演艺": {"keywords": ["演出", "表演", "剧场", "实景", "音乐会", "演唱会", "戏剧", "艺术"], "weight": 2},

    # 特色体验类
    "夜游体验": {"keywords": ["夜游", "夜景", "灯光秀", "夜间", "晚上", "夜晚"], "weight": 1},
    "科普教育": {"keywords": ["科普", "教育", "学习", "知识", "展览", "科技", "探索"], "weight": 1},
    "美食购物": {"keywords": ["美食", "小吃", "夜市", "商业街", "购物", "特产"], "weight": 1}
}

# 更细致的人群映射（增加权重）
audience_keywords = {
    "亲子家庭": {"keywords": ["亲子", "儿童", "家庭", "小朋友", "孩子", "乐园", "动物园", "童话", "互动"], "weight": 2},
    "文化爱好者": {"keywords": ["博物馆", "历史", "文物", "遗迹", "文化", "汉墓", "展览", "艺术", "考古"], "weight": 3},
    "年轻人": {"keywords": ["漂流", "滑雪", "演唱会", "刺激", "夜游", "冒险", "运动", "探险", "极限"], "weight": 2},
    "老年人": {"keywords": ["公园", "寺庙", "慢节奏", "休闲", "散步", "安静", "文化", "历史"], "weight": 2},
    "情侣": {"keywords": ["浪漫", "夜景", "湖景", "约会", "灯光秀", "演出", "艺术", "私密"], "weight": 2},
    "学生团体": {"keywords": ["学习", "教育", "科普", "历史", "文化", "探索", "知识", "研学"], "weight": 3},
    "摄影爱好者": {"keywords": ["风景", "景观", "摄影", "取景", "美景", "自然"], "weight": 1}
}


def classify_attraction(row):
    name = str(row['名称'])
    intro = str(row['introduction'])
    text = (name + " " + intro).lower()

    # 初始化主题和人群权重
    theme_scores = defaultdict(int)
    audience_scores = defaultdict(int)

    # 计算主题得分
    for theme, data in theme_keywords.items():
        for kw in data["keywords"]:
            if kw.lower() in text:
                theme_scores[theme] += data["weight"]

    # 计算人群得分
    for group, data in audience_keywords.items():
        for kw in data["keywords"]:
            if kw.lower() in text:
                audience_scores[group] += data["weight"]

    # 特殊规则增强（确保博物馆和文化遗迹被准确分类）
    if "博物馆" in name or "陈列馆" in name or "纪念馆" in name:
        theme_scores["博物馆"] += 5  # 极大权重
        audience_scores["文化爱好者"] += 3
        audience_scores["学生团体"] += 2

    if "汉墓" in name or "楚王" in name or "汉文化" in name:
        theme_scores["两汉文化"] += 5
        theme_scores["历史遗迹"] += 3
        audience_scores["文化爱好者"] += 3

    # 处理开放时间特征
    if isinstance(row['opening_hours'], str):
        opening_hours = row['opening_hours'].lower()
        if "夜" in opening_hours or "晚上" in opening_hours or "18:00" in opening_hours:
            theme_scores["夜游体验"] += 2
            audience_scores["情侣"] += 1
            audience_scores["年轻人"] += 1

    # 选择得分最高的主题和人群（最多3个）
    top_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_audience = sorted(audience_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    # 过滤低分项（阈值可根据需要调整）
    themes = [theme for theme, score in top_themes if score >= 2]
    audience = [group for group, score in top_audience if score >= 2]

    # 确保至少有一个分类
    if not themes:
        if any(kw in name.lower() for kw in ["公园", "湖", "山"]):
            themes = ["自然风光"]
        elif any(kw in name.lower() for kw in ["博物馆", "纪念馆"]):
            themes = ["博物馆"]
        else:
            themes = ["其他"]

    if not audience:
        if "乐园" in name or "欢乐" in name:
            audience = ["亲子家庭"]
        elif "滑雪" in name or "漂流" in name:
            audience = ["年轻人"]
        else:
            audience = ["通用"]

    # 对博物馆和文化类景点进行最终确认
    if "博物馆" in themes and "文化爱好者" not in audience:
        audience.append("文化爱好者")
    if "两汉文化" in themes and "文化爱好者" not in audience:
        audience.append("文化爱好者")

    return {
        "主题": ", ".join(themes),
        "适合人群": ", ".join(audience),
        "主题详情": str(dict(top_themes)),  # 添加详细得分用于调试
        "人群详情": str(dict(top_audience))  # 添加详细得分用于调试
    }


# 应用分类函数
classification_results = df.apply(classify_attraction, axis=1, result_type='expand')
df = pd.concat([df, classification_results], axis=1)

# 选择要输出的列
output_columns = ['名称', '主题', '适合人群', '主题详情', '人群详情', '评分', '评论数', '信息', 'opening_hours']
result_df = df[output_columns]

# 保存结果
result_df.to_csv('classified_attractions_improved.csv', index=False, encoding='utf-8-sig')

print("分类完成，结果已保存到 classified_attractions_improved.csv")
print("\n文化类景点示例：")
print(result_df[result_df['主题'].str.contains('博物馆|两汉文化|历史遗迹')].head(10))