import csv
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

# -------------------------- 配置参数 --------------------------
KEYWORD = "it"
CITY_URL = "https://wh.58.com/"
PAGES = 3
OUTPUT_FILE = "58job_it.csv"
DRIVER_PATH = "./chromedriver"

SEARCH_URL_TEMPLATE = (
    "https://wh.58.com/job/"
    "?key=it"
    "&final=1&jump=1&page={}"
)

# -------------------------- 数据清洗 --------------------------
def clean_salary(salary_text):
    if not salary_text:
        return ""
    salary_text = salary_text.strip()
    if "面议" in salary_text or salary_text == "":
        return "面议"
    nums = re.findall(r"[\d.]+", salary_text)
    if len(nums) >= 2:
        low = round(float(nums[0]))
        high = round(float(nums[1]))
        if "万" in salary_text:
            low = int(low * 10)
            high = int(high * 10)
        return f"{low}k-{high}k"
    elif len(nums) == 1:
        val = round(float(nums[0]))
        if "万" in salary_text:
            val = int(val * 10)
        return f"{val}k"
    else:
        return salary_text

def clean_education(edu_text):
    return edu_text.strip() if edu_text and edu_text.strip() else "不限"

# -------------------------- 主流程 --------------------------
def main():
    chrome_options = Options()
    # chrome_options.add_argument('--headless')

    service = Service(executable_path=DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.maximize_window()

    try:
        print("正在打开58同城武汉首页...")
        driver.get(CITY_URL)
        input(">>> 请在浏览器中完成登录（如扫码/账号），登录成功后按回车键继续...")

        print(f"正在跳转到“{KEYWORD.upper()}”岗位搜索页...")
        search_url = SEARCH_URL_TEMPLATE.format(1)
        driver.get(search_url)
        time.sleep(3)

        all_jobs = []

        for page in range(1, PAGES + 1):
            print(f"正在爬取第 {page} 页...")
            if page > 1:
                url = SEARCH_URL_TEMPLATE.format(page)
                driver.get(url)
                time.sleep(3)

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "li.job_item"))
                )
            except Exception as e:
                print(f"第 {page} 页加载超时或未找到岗位列表，跳过。错误：{e}")
                continue

            for _ in range(3):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)

            soup = BeautifulSoup(driver.page_source, "lxml")
            job_items = soup.select("li.job_item")
            page_data = []

            for item in job_items:
                title_el = item.select_one(".job_name")
                title = title_el.text.strip() if title_el else ""

                comp_el = item.select_one(".comp_name")
                company = comp_el.text.strip() if comp_el else ""

                salary_el = item.select_one(".job_salary")
                salary = clean_salary(salary_el.text) if salary_el else ""

                edu_el = item.select_one(".job_edu")
                education = clean_education(edu_el.text) if edu_el else "不限"

                area_el = item.select_one(".job_area")
                area = area_el.text.strip() if area_el else ""

                link_el = item.select_one("a")
                link = link_el.get("href") if link_el else ""

                # 合并工作地点和岗位名称为一个字段
                location_job = f"{area}/{title}" if area else title

                page_data.append({
                    "工作地点/工作岗位": location_job,
                    "公司名称": company,
                    "薪资": salary,
                    "学历要求": education,
                    "详情链接": link
                })

            print(f"  本页提取到 {len(page_data)} 条岗位")
            all_jobs.extend(page_data)
            time.sleep(2)

        # 去重（依据合并后的地点+岗位组合，以及公司名称）
        seen = set()
        unique_jobs = []
        for job in all_jobs:
            key = (job["工作地点/工作岗位"], job["公司名称"])
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)

        # 保存CSV（字段顺序按新列）
        field_names = ["工作地点/工作岗位", "公司名称", "薪资", "学历要求", "详情链接"]
        with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(unique_jobs)

        print(f"\n爬取完成！共获取 {len(all_jobs)} 条原始数据，去重后保留 {len(unique_jobs)} 条。")
        print(f"结果已保存至：{OUTPUT_FILE}")

    except Exception as e:
        print(f"程序出现异常：{e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()