import datetime
import json
import re
import time
from collections import defaultdict
from functools import partial, reduce, wraps
from math import sqrt
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    ParamSpec,
    Sequence,
    Iterable,
    TypeGuard,
    TypeVar,
    cast,
)

import akshare as ak
import lightgbm as lgb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import polars as pl
import statsmodels.api as sm
import tushare as ts
import zhplot
from backtesting import Backtest, Strategy
from polars._typing import PolarsDataType
from scipy import stats
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tushare.pro.client import DataApi

pc = pl.col

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

def str_to_date(s: str) -> datetime.date:
    table = str.maketrans({
        "年": "-", "月": "-", "日": ""
    })
    s = s.translate(table)
    if s.endswith('活跃中'):
        return datetime.datetime.now().date()
    return datetime.datetime.strptime(s, '%Y-%m-%d').date()

def str_to_speed(s: str) -> float:
    if s in ['风力不详', '未标明']:
        return 0.0
    m1 = re.match(r'每小时(\d+)公里', s)
    m2 = re.match(r'(\d+).*km/h', s)
    m = m1 or m2
    if m:
        return float(m.group(1))
    else:
        raise ValueError(f'未知的风速: {s}')

def str_to_pressure(s: str) -> float:
    if s == '风力不详':
        return 0.0
    s.replace(',', '')
    m = re.match(r'(\d+).*?(百帕|hPa)', s)
    if m:
        return float(m.group(1))
    else:
        raise ValueError(f'未知的压力: {s}')

def str_to_loss(s: str) -> float:
    if s == '无':
        return 0.0
    if m := re.search(r'(\d+\.\d+)亿', s):
        return float(m.group(1)) * 100000000
    if m := re.search(r'(\d+\.\d+)万', s):
        return float(m.group(1)) * 10000
    else:
        return 0

def load_typhoon_df(p: Path):
    raw_tdfg_df = pl.read_excel(p)
    tdfg_rows = []
    replace_table = str.maketrans({
        '-': '－',
        '–': '－',
        '(': '（',
        ')': '）',
        ' ': '',
    })
    for row in raw_tdfg_df.iter_rows(named=True):
        date = row['持续日期'].translate(replace_table)
        # 替换掉括号里的所有内容为空
        dates = re.sub(r'（.*）', '', date).split('－')
        assert len(dates) == 2
        if '年' not in dates[0]:
            dates[0] = f'{row['年份']}年{dates[0]}'
        if '年' not in dates[1]:
            dates[1] = f'{row['年份']}年{dates[1]}'

        start_date = str_to_date(dates[0])
        end_date = str_to_date(dates[1])
        
        speed = str_to_speed(row['持续风速'])
        pressure = str_to_pressure(row['气压'])
        loss = str_to_loss(row['损失'])
        death = row['死亡人数']
        region = row['影响地区']
        
        tdfg_rows.append({
            '风暴名称': row['风暴名称'],
            'year': row['年份'],
            'start_date': start_date,
            'end_date': end_date,
            'speed': speed,
            'pressure': pressure,
            'loss': loss,
            'death': death,
            'region': region,
        })
    tdfg_df = pl.DataFrame(tdfg_rows)
    return tdfg_df


def as_date(s: Any) -> datetime.date:
    assert isinstance(s, datetime.date)
    return s

def calc_china_score(region: str) -> float:
    r = 0
    if '中国大陆' in region:
        # r += 0.5
        if m := re.search(r'(?<=中国大陆)（(.+)）', region):
            regions = m.group(1).split('、')
            r += len(regions) * 0.1
    else:
        # for hua in ['华南', '华东', '华中']:
        #     if hua in region:
        #         r += 0.5
        if '华' in region:
            r += 1.0

    if '台湾' in region:
        r += 0.5
    if '香港' in region:
        r += 0.2
    if '澳门' in region:
        r += 0.2
    return r

def flat_typhoon_df(df: pl.DataFrame) -> pl.DataFrame:
    '''返回一个包含每一天台风信息的df, 包含如下列: 
    
    date	tdfg_name	count	speed	pressure	loss	death	china_score
    '''
    tdfg_dates: dict[datetime.date, list] = defaultdict(list)
    for row in df.rows(named=True):
        date_range = []
        start = row['start_date']
        while start <= row['end_date']:
            date_range.append(start)
            start += datetime.timedelta(days=1)
        
        for d in date_range:
            tdfg_dates[d].append(row)
    tdfg_flat_rows = []

    curr_date = as_date(df['start_date'].min())
    while curr_date <= as_date(df['end_date'].max()):
        today_tdfgs = tdfg_dates[curr_date]
        if not today_tdfgs:
            curr_date += datetime.timedelta(days=1)
            continue

        speeds = np.array([row['speed'] for row in today_tdfgs])
        # speeds = max(speeds)
        speeds = np.sqrt(np.sum(speeds ** 2))
        
        pressures = np.array([row['pressure'] for row in today_tdfgs])
        pressures = max(pressures)
        
        losses = sum([row['loss'] for row in today_tdfgs])
        deaths = sum([row['death'] for row in today_tdfgs if row['death']])

        china_scores = [calc_china_score(row['region']) for row in today_tdfgs]
        china_score = sum(china_scores)
        # print(china_score, [row['region'] for row in today_tdfgs])

        tdfg_flat_rows.append({
            'date': curr_date,
            'tdfg_name': "|".join([row['风暴名称'] for row in today_tdfgs]),
            'count': len(today_tdfgs),
            'speed': speeds,
            'pressure': pressures,
            'loss': losses,
            'death': deaths,
            'china_score': china_score,
        })
        curr_date += datetime.timedelta(days=1)
    tdfg_flat_df = pl.DataFrame(tdfg_flat_rows).filter(
        pc('pressure') > 0,
        pc('speed') > 0,
    ).with_columns(
        loss = pc('loss').log1p(),
        death = pc('death').log1p(),
        pressure = pc('pressure') / 1012, # 1012hPa为标准大气压
        speed = pc('speed') / 112, # 112km/h 是12级风, 是台风的最低标准
    )

    return tdfg_flat_df


def load_baidu_idnex(
    base_dir: Path, 
    *, 
    index_list: Iterable[str]=['台风', '台风实时路径', '台风路径', '暴雨', '暴雨预警'], 
    use_log1p_for_index = True
) -> pl.DataFrame:
    tdfg_baidu_index_dfs: list[pl.DataFrame] = []
    for index in index_list:
        tdfg_baidu_index_dfs.append(
            pl.read_excel(base_dir / f'{index}-百度指数.xlsx').select(
                'date',
                # pc('index').alias(f'index-{index}').log1p(),
                pc('index').alias(f'index-{index}'),
            ).with_columns(
                pc(f'index-{index}').log1p() if use_log1p_for_index else pc(f'index-{index}'),
            )
        )
    tdfg_baidu_index_df = reduce(lambda x, y: x.join(y, on='date', how='left'), tdfg_baidu_index_dfs).select(
        pl.all().exclude('index-台风路径', 'index-台风实时路径'),
        
        ((pc('index-台风路径').exp() + pc('index-台风实时路径').exp()) / 2).log1p().alias('index-台风路径') if use_log1p_for_index else ((pc('index-台风路径') + pc('index-台风实时路径')) / 2).alias('index-台风路径'),
    )
    return tdfg_baidu_index_df
