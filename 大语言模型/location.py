import requests
from bs4 import BeautifulSoup
import csv
import time
from urllib.parse import urljoin

# 基础URL和headers
base_url = "https://you.ctrip.com"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def get_attraction_details(url):
    """获取单个景点的介绍和开放时间"""
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 获取介绍
        intro_title = soup.find('div', class_='moduleTitle', string='介绍')
        intro = ""
        if intro_title:
            intro_content = intro_title.find_next_sibling('div', class_='moduleContent')
            if intro_content:
                intro_div = intro_content.find('div', class_='LimitHeightText')
                intro = intro_div.get_text(strip=True) if intro_div else ""

        # 获取开放时间
        time_title = soup.find('div', class_='moduleTitle', string='开放时间')
        open_time = ""
        if time_title:
            time_content = time_title.find_next_sibling('div', class_='moduleContent')
            open_time = time_content.get_text(strip=True) if time_content else ""

        return {
            'introduction': intro,
            'opening_hours': open_time
        }

    except Exception as e:
        print(f"获取景点详情失败 ({url}): {e}")
        return {
            'introduction': '',
            'opening_hours': ''
        }


def main():
    # 输入和输出文件
    input_csv = 'attractions.csv'
    output_csv = 'attractions_add.csv'

    # 从输入CSV读取景点数据
    try:
        with open(input_csv, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            original_fieldnames = reader.fieldnames
            rows = list(reader)  # 读取所有行

            # 检查是否有 '详情页链接' 列
            if '详情页链接' not in original_fieldnames:
                print("错误：输入CSV文件中没有 '详情页链接' 列")
                return

    except FileNotFoundError:
        print(f"输入文件 {input_csv} 不存在")
        return
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        return

    # 准备输出CSV的字段名（保留原始字段+新增字段）
    new_fieldnames = original_fieldnames + ['introduction', 'opening_hours']

    with open(output_csv, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=new_fieldnames)
        writer.writeheader()

        # 遍历每行数据并获取详情
        for i, row in enumerate(rows, 1):
            url = row['详情页链接']  # 修改为 '详情页链接'
            print(f"正在处理 {i}/{len(rows)}: {url}")

            # 获取详情信息
            details = get_attraction_details(url)

            # 更新行数据
            row.update(details)

            # 写入新行
            writer.writerow(row)

            # 礼貌爬取，添加延迟
            time.sleep(2)  # 2秒间隔

    print(f"数据已保存到 {output_csv}")


if __name__ == '__main__':
    main()