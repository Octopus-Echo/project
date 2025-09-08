import csv
import time
import random
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# 基础配置
BASE_URL = "https://you.ctrip.com"
MAX_RETRIES = 3
PAGE_LOAD_TIMEOUT = 30
USE_SELENIUM = True  # 设置为False则使用requests方式


class CtripSpider:
    def __init__(self):
        self.ua = UserAgent()
        self.headers = {
            'User-Agent': self.ua.random,
            'Referer': BASE_URL,
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }
        self.driver = None
        if USE_SELENIUM:
            self.driver = self._init_selenium()

    def _init_selenium(self):
        """初始化Selenium驱动"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'user-agent={self.ua.random}')

            # 自动下载和管理ChromeDriver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
            return driver
        except Exception as e:
            logging.error(f"初始化Selenium失败: {e}")
            return None

    def get_page_content(self, url):
        """获取页面内容（自动选择方式）"""
        if USE_SELENIUM and self.driver:
            return self._get_page_with_selenium(url)
        else:
            return self._get_page_with_requests(url)

    def _get_page_with_selenium(self, url, retry=0):
        """使用Selenium获取页面"""
        try:
            time.sleep(random.uniform(1, 3))

            # 设置随机User-Agent
            self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": self.ua.random
            })

            logging.info(f"使用Selenium访问: {url}")
            self.driver.get(url)

            # 等待主要内容加载
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'list_mod2')))
            except TimeoutException:
                logging.warning("页面元素加载超时，但继续处理...")

            # 模拟滚动
            for _ in range(2):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(0.5, 1.5))

            return self.driver.page_source

        except Exception as e:
            if retry < MAX_RETRIES:
                logging.warning(f"Selenium访问失败，第{retry + 1}次重试...")
                return self._get_page_with_selenium(url, retry + 1)
            logging.error(f"Selenium获取页面失败: {e}")
            return None

    def _get_page_with_requests(self, url, retry=0):
        """使用requests获取页面"""
        try:
            time.sleep(random.uniform(1, 3))
            logging.info(f"使用Requests访问: {url}")

            # 每次请求使用随机User-Agent
            self.headers['User-Agent'] = self.ua.random
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text

        except Exception as e:
            if retry < MAX_RETRIES:
                logging.warning(f"Requests访问失败，第{retry + 1}次重试...")
                return self._get_page_with_requests(url, retry + 1)
            logging.error(f"Requests获取页面失败: {e}")
            return None

    def parse_attraction_list(self, html):
        """解析景点列表"""
        soup = BeautifulSoup(html, 'html.parser')
        attractions = []
        items = soup.find_all('div', class_='list_mod2')

        if not items:
            logging.warning("未找到景点列表，可能需要更新选择器")
            return attractions

        for item in items:
            try:
                name_tag = item.find('dt').find('a', attrs={'title': True}) if item.find('dt') else None
                name = name_tag['title'] if name_tag and 'title' in name_tag.attrs else (
                    name_tag.get_text(strip=True) if name_tag else "未知名称")

                # 详情页链接
                detail_link = name_tag['href'] if name_tag and 'href' in name_tag.attrs else ""
                # 处理URL
                if detail_link and not detail_link.startswith('http'):
                    if detail_link.startswith('//'):
                        detail_link = 'https:' + detail_link
                    elif detail_link.startswith('/'):
                        detail_link = BASE_URL + detail_link
                    else:
                        detail_link = BASE_URL + '/' + detail_link

                # 评分
                rating_tag = item.find('strong')
                rating = rating_tag.get_text(strip=True) if rating_tag else "无评分"

                # 评论数量
                reviews_tag = item.find('a', class_='recomment')
                reviews = reviews_tag.get_text(strip=True).replace('条点评', '') if reviews_tag else "0"

                # 景点信息
                info_tag = item.find('dd', class_='ellipsis')
                info = info_tag.get_text(strip=True) if info_tag else "无信息"

                # 图片
                img_tag = item.find('img')
                img_url = img_tag['src'] if img_tag and 'src' in img_tag.attrs else "无图片"

                attractions.append({
                    '名称': name,
                    '详情页链接': detail_link,
                    '评分': rating,
                    '评论数': reviews,
                    '信息': info,
                    '图片链接': img_url
                })

            except Exception as e:
                logging.error(f"解析景点出错: {e}")
                continue

        return attractions

    def save_to_csv(self, data, filename='attractions.csv'):
        """保存数据到CSV"""
        if not data:
            logging.warning("没有数据可保存")
            return

        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            logging.info(f"成功保存{len(data)}条数据到{filename}")
        except Exception as e:
            logging.error(f"保存CSV失败: {e}")

    def crawl(self, start_page=1, end_page=3):
        """执行爬取任务"""
        all_attractions = []

        try:
            for page in range(start_page, end_page + 1):
                logging.info(f"开始爬取第{page}页...")
                url = f"{BASE_URL}/sightlist/xuzhou230/s0-p{page}.html"

                html = self.get_page_content(url)
                if not html:
                    continue

                attractions = self.parse_attraction_list(html)
                if attractions:
                    all_attractions.extend(attractions)
                    logging.info(f"第{page}页获取到{len(attractions)}个景点")
                else:
                    logging.warning(f"第{page}页未找到景点数据")

                time.sleep(random.uniform(2, 4))  # 随机延迟

            if all_attractions:
                logging.info(f"总共获取{len(all_attractions)}个景点")
                # 打印前3个示例
                for i, item in enumerate(all_attractions[:3], 1):
                    logging.info(
                        f"示例{i}: {item['名称']} | 评分: {item['评分']} | "
                        f"评论: {item['评论数']} | 信息: {item['信息']}"
                    )
                self.save_to_csv(all_attractions)
            else:
                logging.warning("未获取到任何景点数据")

        except Exception as e:
            logging.error(f"爬取过程中出错: {e}")
        finally:
            if self.driver:
                self.driver.quit()
                logging.info("已关闭浏览器")


if __name__ == "__main__":
    spider = CtripSpider()
    spider.crawl(start_page=1, end_page=8)