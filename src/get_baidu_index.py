# %%
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import zhplot
import numpy as np
import pandas as pd
import polars as pl
import requests
from requests.exceptions import RequestException


def generate_http_headers(credential):
    http_headers = {
        'Cookie': 'BDUSS=' + credential["cookie_BDUSS"],
        'Cipher-Text': credential["cipherText"],
        'Accept': 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://index.baidu.com/v2/main/index.html',
        'Host': 'index.baidu.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    return http_headers


def calculate_yearly_averages(start_date, end_date, data_series):
    # Convert the start and end dates to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days_span = (end - start).days + 1

    # Split the data series into a list and replace empty strings with '0'
    data_points = data_series.split(',')
    data_points = ['0' if point == '' else point for point in data_points]
    data_points = np.array(data_points, dtype=float)

    if days_span <= 366:
        dates = pd.date_range(start, periods=len(data_points))
    else:
        weeks_span = len(data_points)
        dates = pd.date_range(start, periods=weeks_span, freq='W')

    # Create a DataFrame with the dates and data points
    df = pd.DataFrame({'Date': dates, 'Data': data_points})
    df.set_index('Date', inplace=True)

    return df


# 解密
def decrypt(ptbk, index_data):
    n = len(ptbk) // 2
    a = dict(zip(ptbk[:n], ptbk[n:]))
    return "".join([a[s] for s in index_data])


def keywords2json(keyword):
    import json
    # Convert each keyword in each sublist into a dictionary with 'name' and 'wordType'
    converted_keywords = [[
        {"name": keyword, "wordType": 1}
    ]]
    # Convert the list of lists of dictionaries into a JSON string
    json_string = json.dumps(converted_keywords, ensure_ascii=False)
    return json_string



def crawl_request(keyword, startDate, endDate, credential):
    print('正在查询：', keyword, startDate, endDate)
    words = keywords2json(keyword)

    max_retries = 3
    retries = 0

    # https://index.baidu.com/api/SearchApi/index?area=0&word=[[%7B%22name%22:%22%E5%8F%B0%E9%A3%8E%22,%22wordType%22:1%7D]]&startDate=2025-02-01&endDate=2025-09-27

    # http://index.baidu.com/api/SearchApi/index?area=0&word=[[{"name": "台风", "wordType": 1}]]&startDate=2025-01-01&endDate=2025-12-31

    while retries < max_retries:
        url = f'http://index.baidu.com/api/SearchApi/index?area=0&word={words}&startDate={startDate}&endDate={endDate}'
        print(url)
        rsp = requests.get(url, headers=generate_http_headers(credential), timeout=10).json()

        # 获取解密秘钥
        data = rsp['data']['userIndexes']
        uniqid = rsp['data']['uniqid']
        url = f'https://index.baidu.com/Interface/ptbk?uniqid={uniqid}'
        ptbk = requests.get(url, headers=generate_http_headers(credential), timeout=10).json()['data']

        # 数据解密
        res = []
        for i in range(len(data)):
            index_data = decrypt(ptbk, data[i]['all']['data'])
            df = calculate_yearly_averages(startDate, endDate, index_data)

            res.append(df)
        return res

    if retries == max_retries:
        print(f'请求失败次数过多，已达到最大重试次数{max_retries}，跳过')
        return None

# %%
credential = {
    "cookie_BDUSS": '你的BDUSS',
    "cipherText": "你的cipherText",
}

for keyword in ['台风', '台风实时路径', '台风路径', '暴雨', '暴雨预警']:
    output_p = Path(__file__).parent / 'data' / 'raw' / 'baidu_index' / f'{keyword}-百度指数.xlsx'
    output_p.parent.mkdir(parents=True, exist_ok=True)
    # print(output_p, output_p.exists())

    if output_p.exists():
        continue

    dfs = []
    for year in range(2017, 2026):
        r = crawl_request(keyword, f'{year}-01-01', f'{year}-12-31', credential)
        if r is None:
            print(f'{year} 查询失败')
            continue
        dfs.extend(r)
        time.sleep(random.randint(2, 4) / 2)
    df = pl.from_pandas(pd.concat(dfs).sort_index(), include_index=True).select(
        date=pl.col('Date').dt.date(),
        index='Data',
    )
    df.write_excel(output_p)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['date'], df['index'])
    ax.set_title(keyword)
    ax.set_xlabel('Date')
    ax.set_ylabel('Index')
    fig.savefig(output_p.with_suffix('.png'))
    plt.close(fig)

# %%
