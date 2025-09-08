from flask import Flask, request, render_template
from tourism_recommender import TourismRecommender
from datetime import datetime, timedelta

app = Flask(__name__)
recommender = TourismRecommender()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取表单数据
        themes = request.form.getlist('themes')
        audiences = request.form.getlist('audiences')
        target_area = request.form.get('area', '')
        start_date = request.form.get('start_date', '')
        end_date = request.form.get('end_date', '')
        top_n = request.form.get('top_n')  # 改为可选参数

        # 处理日期区间
        date_range = []
        if start_date:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            if end_date:
                end = datetime.strptime(end_date, '%Y-%m-%d').date()
                if end < start:
                    end = start  # 如果结束日期早于开始日期，则设为同一天
                date_range = [start + timedelta(days=i) for i in range((end - start).days + 1)]
            else:
                date_range = [start]

        # 获取推荐结果
        recs = recommender.recommend(
            themes=themes,
            audiences=audiences,
            target_area=target_area,
            date_range=date_range if date_range else None,
            top_n=int(top_n) if top_n else None  # 如果未指定top_n，则传None
        )

        # 生成旅游计划
        plan = recommender.generate_travel_plan(recs, date_range if date_range else None)
    else:
        plan = None

    return render_template('index.html',
                         themes=sorted(recommender.themes),
                         audiences=sorted(recommender.audiences),
                         areas=sorted(recommender.areas),
                         plan=plan)

if __name__ == '__main__':
    app.run(debug=True, port=5000)