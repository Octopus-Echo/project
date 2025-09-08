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

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ctrip_spider.log'), logging.StreamHandler()]
)


class DynamicCtripSpider:
    def __init__(self):
        """初始化动态爬虫"""
        # 先定义所有属性
        self.base_url = "https://you.ctrip.com"
        self.max_retries = 3
        self.page_load_timeout = 30

        self.ua = UserAgent()
        self.headers = {
            'User-Agent': self.ua.random,
            'Referer': self.base_url + '/',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }

        # 最后初始化Selenium
        self.driver = self._init_selenium()

    def _init_selenium(self):
        """初始化Selenium浏览器驱动"""
        try:
            chrome_options = Options()
            # 开发时可注释掉headless模式以便观察
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'user-agent={self.ua.random}')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')

            # 禁用自动化控制标志
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)

            # 设置下载管理器
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)

            # 修改navigator.webdriver标志
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """
            })

            driver.set_page_load_timeout(self.page_load_timeout)
            return driver
        except Exception as e:
            logging.error(f"初始化Selenium失败: {e}")
            raise

    def human_like_scroll(self, scroll_times=3):
        """模拟人类滚动行为"""
        for i in range(scroll_times):
            # 随机滚动距离和间隔
            scroll_px = random.randint(500, 1000)
            self.driver.execute_script(f"window.scrollBy(0, {scroll_px});")
            # 随机等待时间
            time.sleep(random.uniform(1.0, 2.5))

            # 随机横向滚动
            if random.random() > 0.7:
                h_scroll = random.randint(-100, 100)
                self.driver.execute_script(f"window.scrollBy({h_scroll}, 0);")

    def wait_for_content_load(self, timeout=20):
        """等待主要内容加载"""
        try:
            # 等待直到至少有一个景点卡片加载出来
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[class*="sightItemCard"]'))
            )

            # 额外等待内容稳定
            time.sleep(random.uniform(1.0, 2.0))

        except Exception as e:
            logging.warning(f"等待内容加载超时: {e}")
            # 即使超时也继续，可能有部分内容

    def get_page_content(self, url, retry=0):
        """获取页面内容"""
        try:
            if retry > 0:
                logging.info(f"第{retry}次重试获取页面: {url}")

            # 随机延迟
            time.sleep(random.uniform(2, 5))

            # 设置随机User-Agent
            self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": self.ua.random
            })

            logging.info(f"访问URL: {url}")
            self.driver.get(url)

            # 等待内容加载
            self.wait_for_content_load()

            # 模拟人类滚动行为
            self.human_like_scroll(scroll_times=random.randint(2, 4))

            # 检查是否有"加载更多"按钮
            self.click_load_more()

            # 再次滚动确保所有内容加载
            self.human_like_scroll(scroll_times=1)

            return self.driver.page_source

        except Exception as e:
            if retry < self.max_retries:
                logging.warning(f"获取页面失败，第{retry + 1}次重试... 错误: {str(e)}")
                return self.get_page_content(url, retry + 1)
            logging.error(f"获取页面失败: {e}")
            return None

    def click_load_more(self):
        """尝试点击'加载更多'按钮"""
        try:
            # 查找所有可能的加载更多按钮
            buttons = self.driver.find_elements(By.XPATH,
                                                "//div[contains(@class, 'load') and contains(text(), '加载更多')]")
            for btn in buttons:
                try:
                    if btn.is_displayed():
                        btn.click()
                        logging.info("点击了'加载更多'按钮")
                        time.sleep(random.uniform(2.0, 3.5))
                        break
                except:
                    continue
        except Exception as e:
            logging.debug(f"查找加载更多按钮时出错: {e}")

    def parse_attraction_list(self, html):
        """解析景点列表页面"""
        soup = BeautifulSoup(html, 'html.parser')
        attractions = []

        # 使用更灵活的选择器
        items = soup.find_all('div', class_=lambda x: x and 'sightItemCard' in x)

        if not items:
            logging.warning("未找到景点列表，可能需要更新选择器")
            return attractions

        for item in items:
            try:
                # 1. 景点名称和链接
                name_tag = item.find('a', href=True)
                name = name_tag.get_text(strip=True) if name_tag else "未知名称"
                href = name_tag['href'] if name_tag else ""

                # 处理详情链接
                if href.startswith('http'):
                    detail_link = href
                elif href.startswith('//'):
                    detail_link = 'https:' + href
                elif href.startswith('/'):
                    detail_link = self.base_url + href
                else:
                    detail_link = href

                # 2. 景点等级
                level = "无等级信息"
                level_tags = item.find_all('span', class_=lambda x: x and 'level' in x)
                for tag in level_tags:
                    if tag.get_text(strip=True):
                        level = tag.get_text(strip=True)
                        break

                # 3. 评分信息
                heat_score = "无热度评分"
                user_score = "无评分"
                score_spans = item.find_all('span', class_=lambda x: x and 'score' in x)
                for span in score_spans:
                    text = span.get_text(strip=True)
                    if '分' in text:
                        user_score = text.replace('分', '')
                    elif text.replace('.', '').isdigit():
                        heat_score = text

                # 4. 评论数量
                comment_num = "0"
                comment_tags = item.find_all(string=lambda text: '条点评' in text if text else False)
                if comment_tags:
                    comment_num = comment_tags[0].split('条')[0]

                # 5. 服务标签
                service_tags = []
                tag_divs = item.find_all('div', class_=lambda x: x and 'tag' in x)
                for div in tag_divs:
                    tags = [t.get_text(strip=True) for t in div.find_all('span') if t.get_text(strip=True)]
                    service_tags.extend(tags)

                # 6. 价格信息
                price = "无价格信息"
                price_divs = item.find_all('div', class_=lambda x: x and 'price' in x)
                for div in price_divs:
                    if '免费' in div.get_text():
                        price = "免费"
                        break
                    price_span = div.find('span')
                    if price_span:
                        price = price_span.get_text(strip=True)
                        if price:
                            break

                # 添加到结果列表
                attractions.append({
                    '景点名称': name,
                    '详情链接': detail_link,
                    '景点等级': level,
                    '热度评分': heat_score,
                    '用户评分': user_score,
                    '评论数量': comment_num,
                    '服务标签': '|'.join(service_tags),
                    '价格': price
                })

            except Exception as e:
                logging.error(f"解析景点出错: {e}")
                continue

        return attractions

    def crawl_pages(self, start_page=1, end_page=8):
        """爬取多个页面"""
        all_attractions = []

        for page in range(start_page, end_page + 1):
            url = f"{self.base_url}/sight/xuzhou230/s0-p{page}.html"
            logging.info(f"开始爬取第 {page} 页: {url}")

            html = self.get_page_content(url)
            if not html:
                logging.warning(f"第 {page} 页获取失败，跳过...")
                continue

            # 保存页面快照用于调试
            with open(f'debug_page_{page}.html', 'w', encoding='utf-8') as f:
                f.write(html)

            attractions = self.parse_attraction_list(html)
            if attractions:
                all_attractions.extend(attractions)
                logging.info(f"第 {page} 页获取到 {len(attractions)} 个景点")
            else:
                logging.warning(f"第 {page} 页未解析到景点数据")

            # 随机延迟
            time.sleep(random.uniform(3, 8))

        return all_attractions

    def save_to_csv(self, data, filename='xuzhou_attractions.csv'):
        """保存数据到CSV"""
        if not data:
            logging.warning("没有数据可保存")
            return False

        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            logging.info(f"成功保存 {len(data)} 条数据到 {filename}")
            return True
        except Exception as e:
            logging.error(f"保存CSV失败: {e}")
            return False

    def run(self):
        """运行爬虫"""
        try:
            # 爬取数据
            attractions = self.crawl_pages(start_page=1, end_page=8)

            if attractions:
                # 去重处理
                unique_attractions = []
                seen = set()
                for item in attractions:
                    identifier = (item['景点名称'], item['详情链接'])
                    if identifier not in seen:
                        seen.add(identifier)
                        unique_attractions.append(item)

                logging.info(f"共获取 {len(unique_attractions)} 个唯一景点")

                # 保存数据
                self.save_to_csv(unique_attractions)

                # 打印示例
                for i, item in enumerate(unique_attractions[:3], 1):
                    logging.info(
                        f"示例 {i}: {item['景点名称']} | 评分: {item['用户评分']} | "
                        f"价格: {item['价格']} | 评论: {item['评论数量']}"
                    )
            else:
                logging.warning("未获取到任何景点数据")

        finally:
            if self.driver:
                self.driver.quit()
                logging.info("浏览器已关闭")


if __name__ == "__main__":
    spider = DynamicCtripSpider()
    spider.run()
