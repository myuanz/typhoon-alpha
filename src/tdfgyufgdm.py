# %%
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
    TypeGuard,
    TypeVar,
    cast,
)
from backtesting import Backtest, Strategy

import akshare as ak
import lightgbm as lgb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import plotly.express as px
import polars as pl
import statsmodels.api as sm
import tushare as ts
import yfinance as yf
import zhplot
from polars._typing import PolarsDataType
from scipy import stats
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

pc = pl.col

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")  # 用于方法的 self/type 位置

data_dir = Path(__file__).parent / 'data' / 'raw'
result_dir = Path(__file__).parent / 'data' / f'result-{datetime.datetime.now():%Y%m%d-%H%M%S}'
# result_dir = Path(__file__).parent / 'data' / 'result-20250930-185854'

data_dir.mkdir(parents=True, exist_ok=True)
result_dir.mkdir(parents=True, exist_ok=True)

def to_pl(func: Callable[P, pd.DataFrame]) -> Callable[P, pl.DataFrame]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> pl.DataFrame:
        return pl.from_pandas(func(*args, **kwargs))
    return wrapper

def cache_df(name: str | Callable[[tuple, dict], str], ext: Literal['csv', 'parquet'] = 'csv', force: bool = False, schema_overrides: Mapping[str, PolarsDataType] | Sequence[PolarsDataType] | None | None = None):
    def decorator(func: Callable[P, pl.DataFrame | pd.DataFrame]) -> Callable[P, pl.DataFrame]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> pl.DataFrame:
            if callable(name):
                name_str = name(args, kwargs)
            else:
                name_str = name

            p = data_dir / f'{name_str}.{ext}'
            if p.exists() and not force:
                read_func = {
                    'csv'    : lambda p: pl.read_csv(p, schema_overrides=schema_overrides),
                    'parquet': pl.read_parquet,
                }[ext]
                return read_func(p)

            df = func(*args, **kwargs)
            if isinstance(df, pd.DataFrame):
                df = pl.from_pandas(df)

            write_func = {
                'csv'    : pl.DataFrame.write_csv,
                'parquet': pl.DataFrame.write_parquet,
            }[ext]
            write_func(df, p)
            return df
        return wrapper
    return decorator
# %%
pro = ts.pro_api()
# %%
all_stock_df = cache_df('all_stock_df', schema_overrides={
    'symbol': pl.String,
    'list_date': pl.String,
})(to_pl(pro.stock_basic))()
all_stock_df
# %%
stock_df = pl.read_excel(data_dir / 'wind_input_stocks.xlsx').with_columns(
    symbol=pc('symbol').str.replace(r'\D+', '')
)
raw_df_length = len(stock_df)
stock_df
# %%
stock_df = stock_df.join(all_stock_df, on='symbol', how='left').filter(
    pl.col('ts_code').is_null().not_()
)

if len(stock_df) != raw_df_length:
    print(f'{len(stock_df)} != {raw_df_length}')
if (stock_df['name'] == stock_df['name_right']).all():
    print(f'{stock_df["name"]} == {stock_df["name_right"]}')

stock_df.drop_in_place('name_right')
stock_df.write_excel(result_dir / '10.wind_stock_df.xlsx')
stock_df
# %%
# stock_df = all_stock_df
# %%
@cache_df(lambda args, kwargs: f'adj-no/daily_data_{args[0]}_{args[1]}_{args[2]}', schema_overrides={
    'trade_date': pl.String,
    'date': pl.Date,
}, ext='parquet')
def fetch_daily_data(ts_code: str, start_date: str, end_date: str, is_index: bool = False) -> pl.DataFrame:
    if is_index:
        func = pro.index_daily
    else:
        func = partial(ts.pro_bar, adj='hfq', asset='E')
    return pl.from_pandas(func(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
    )).sort('trade_date').with_columns(
        date=pl.col('trade_date').str.to_date('%Y%m%d'),
    )
fetch_daily_data('000001.SZ', '20050101', '20250926')
fetch_daily_data('838670.BJ', '20050101', '20250926')

# %%
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

raw_tdfg_df = pl.read_excel(data_dir / '台风.xlsx')
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
    print(row['风暴名称'], start_date, end_date)
    
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
tdfg_df.write_excel(result_dir / '20.台风-norm.xlsx')

tdfg_df
# %%

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


tdfg_dates: dict[datetime.date, list] = defaultdict(list)
for row in tdfg_df.rows(named=True):
    tdfg_name = f'{row['风暴名称']}-{row["year"]}'
    date_range = []
    start = row['start_date']
    while start <= row['end_date']:
        date_range.append(start)
        start += datetime.timedelta(days=1)
    
    for d in date_range:
        tdfg_dates[d].append(row)
tdfg_flat_rows = []

curr_date = as_date(tdfg_df['start_date'].min())
while curr_date <= as_date(tdfg_df['end_date'].max()):
    today_tdfgs = tdfg_dates[curr_date]
    if not today_tdfgs:
        curr_date += datetime.timedelta(days=1)
        continue

    speeds = np.array([row['speed'] for row in today_tdfgs])
    # speeds = (speeds ** 2).sum() ** 0.5
    # speeds = speeds.sum()
    speeds = max(speeds)
    
    pressures = np.array([row['pressure'] for row in today_tdfgs])
    # pressures = (pressures ** 2).sum() ** 0.5
    # pressures = pressures.sum()
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
index_list = ['台风', '台风实时路径', '台风路径', '暴雨', '暴雨预警']
use_log1p_for_index = True

tdfg_baidu_index_dfs: list[pl.DataFrame] = []
for index in index_list:
    tdfg_baidu_index_dfs.append(
        pl.read_excel(data_dir / 'baidu_index' / f'{index}-百度指数.xlsx').select(
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
tdfg_baidu_index_df
# %%

tdfg_flat_df = tdfg_baidu_index_df.join(tdfg_flat_df, on='date', how='left').fill_null(0).drop_nulls()
tdfg_flat_df
# %%

start_date = datetime.date(2017, 5, 1)

# final_dfs = []

# for row in tqdm(stock_df.iter_rows(named=True), total=len(stock_df)):
#     r = fetch_daily_data(row['ts_code'], '20050101', '20250926').filter(
#         pc('date') >= start_date
#     ).with_columns(
#         norm_open=pl.col('open') / pl.col('open').first(),
#         norm_high=pl.col('high') / pl.col('high').first(),
#         norm_low=pl.col('low') / pl.col('low').first(),
#         norm_close=pl.col('close') / pl.col('close').first(),
#     )
#     final_dfs.append(r)
# final_df: pl.DataFrame = pl.concat(final_dfs)
# '''eg: 
# ts_code	trade_date	open	high	low	close	pre_close	change	pct_chg	vol	amount	date	norm_open	norm_high	norm_low	norm_close
# str	str	f64	f64	f64	f64	f64	f64	f64	f64	f64	date	f64	f64	f64	f64
# "300904.SZ"	"20240506"	46.07	46.8	45.68	46.47	45.22	1.25	2.7643	12871.13	59576.707	2024-05-06	1.0	1.0	1.0	1.0
# "300904.SZ"	"20240507"	46.56	48.48	45.65	47.91	46.47	1.44	3.0988	25654.84	121290.769	2024-05-07	1.010636	1.035897	0.999343	1.030988
# "300904.SZ"	"20240508"	47.4	48.64	46.5	46.88	47.91	-1.03	-2.1499	18443.0	87841.781	2024-05-08	1.028869	1.039316	1.017951	1.008823
# "300904.SZ"	"20240509"	46.43	49.79	46.43	48.69	46.88	1.81	3.8609	23110.17	111337.986	2024-05-09	1.007814	1.063889	1.016419	1.047773
# "300904.SZ"	"20240510"	51.04	55.88	49.26	49.36	48.69	0.67	1.3761	44095.41	230513.977	2024-05-10	1.107879	1.194017	1.078371	1.062191
# …	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…
# "603063.SH"	"20250922"	33.6	35.18	33.03	34.12	33.69	0.43	1.2763	279140.95	946274.25	2025-09-22	1.699545	1.733859	1.670713	1.689109
# "603063.SH"	"20250923"	34.4	35.18	33.2	34.14	34.12	0.02	0.0586	283919.16	968189.759	2025-09-23	1.74001	1.733859	1.679312	1.690099
# "603063.SH"	"20250924"	33.7	33.93	32.83	33.82	34.14	-0.32	-0.9373	287430.2	958228.651	2025-09-24	1.704603	1.672252	1.660597	1.674257
# "603063.SH"	"20250925"	33.59	34.74	33.52	34.43	33.82	0.61	1.8037	333190.69	1.1423e6	2025-09-25	1.699039	1.712173	1.695498	1.704455
# "603063.SH"	"20250926"	34.35	35.5	34.0	34.0	34.43	-0.43	-1.2489	420469.1	1.4657e6	2025-09-26	1.737481	1.74963	1.719777	1.683168
# '''
# final_df.write_parquet(result_dir / '30.final_df.parquet')
# final_df
# # %%
# merged_final_df: pl.DataFrame = final_df.group_by('date').agg(
#     pl.col('norm_open').mean().alias('norm_open'),
#     pl.col('norm_high').mean().alias('norm_high'),
#     pl.col('norm_low').mean().alias('norm_low'),
#     pl.col('norm_close').mean().alias('norm_close'),
#     amount=pl.col('amount').sum().alias('amount'),
# ).sort('date')
# '''eg: 
# date	norm_open	norm_high	norm_low	norm_close	amount
# str	date	f64	f64	f64	f64	f64
# 2025-07-01	0.848385	0.839612	0.854015	0.866221	138676.933
# 2025-05-08	0.921689	0.891456	0.912692	0.913225	232141.031
# 2025-02-17	1.068457	1.048379	1.049526	1.047967	140458.723
# 2025-08-06	0.895799	0.886076	0.877922	0.870396	292780.901
# 2024-11-01	0.929671	0.906617	0.903158	0.890203	175510.124
# '''
# merged_final_df.write_excel(result_dir / '40.merged_final_df.xlsx')
# merged_final_df
# %%
上证指数df = fetch_daily_data('000001.SH', '20050101', '20250926', is_index=True)
上证指数df
# %%
pct_expr = pl.col('close').pct_change()

uhvg_pct_df = 上证指数df.select(
    'date',
    return_pct='pct_chg',
).filter(
    pc('date') >= start_date,
    # pc('return_pct').is_null().not_()
)
uhvg_pct_df

# %%
start_date = datetime.date(2017, 5, 1)

return_pct_dfs: list[pl.DataFrame] = []

total_results = {}

for stock_index, curr_stock_row in tqdm(enumerate(stock_df.iter_rows(named=True)), total=len(stock_df)):

    if 'get_ipython' in globals():
        if curr_stock_row['name'] != '日出东方':
            continue
    print(curr_stock_row)
    curr_stock_dir = result_dir / f'{curr_stock_row["name"]}'
    if (curr_stock_dir / f'20-回归.txt').exists():
        continue
    
    curr_stock_dir.mkdir(parents=True, exist_ok=True)

    curr_stock_ohlcv_df = fetch_daily_data(curr_stock_row['ts_code'], '20050101', '20250926').filter(
        pc('date') >= start_date
    ).select(
        'ts_code',
        'date',
        pl.col('open').alias('Open'),
        pl.col('high').alias('High'),
        pl.col('low').alias('Low'),
        pl.col('close').alias('Close'),
        return_pct='pct_chg',
    )
    # return_pct_dfs.append(r)

    fgdm_pct_df = curr_stock_ohlcv_df.filter(
        pc('return_pct').is_null().not_(),
        pc('date') >= start_date,
    ).group_by('date').agg(
        pl.col('return_pct').mean().alias('return_pct'),
    ).sort('date')
    fgdm_pct_df

# %%
    pct_df = fgdm_pct_df.join(uhvg_pct_df, on='date').select(
        'date',
        fgdm='return_pct',
        uhvg='return_pct_right',
    ).join(tdfg_flat_df, on='date', how='left').sort('date')
    pct_df
# %%
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(pct_df['date'], pct_df['fgdm'])
    # axes[0].plot(pct_df['date'], pct_df['uhvg'])
    axes[0].twinx().plot(pct_df['date'], pct_df['index-台风'], color='gray', linestyle='--', alpha=0.5)
    axes[1].scatter(pct_df['uhvg'], pct_df['fgdm'])
    axes[1].set_xlabel('上证指数 pct')
    axes[1].set_ylabel('风电 pct')
    fig.savefig(curr_stock_dir / f'10-相关性.png')

# %%
    df = pct_df.filter(
        # pc('date') >= datetime.date(2024, 5, 1),
        # pc('speed') > 0.7,
    ).to_pandas().copy()
    print(f'{pct_df.shape=} -> {df.shape=}')
    df['date'] = pd.to_datetime(df['date'])

    ty = tdfg_df.to_pandas().copy()
    ty['start_ts'] = pd.to_datetime(ty['start_date'])
    ty['end_ts']   = pd.to_datetime(ty['end_date'])

# %% 市场模型回归：fgdm ~ α + β*uhvg  → 拟合残差 ε_t
    X = sm.add_constant(df['uhvg'])
    y = df['fgdm']
    mkt_model = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')  # 鲁棒标准误
    mkt_r2_uhvg_only = mkt_model.rsquared
    df['resid'] = mkt_model.resid  # 残差序列（超额收益的“无法由市场解释”的部分）

    try:
        summary = mkt_model.summary()
    except Exception as e:
        print(f'{e}')
        continue

    with open(curr_stock_dir / f'20-回归.txt', 'w') as f:
        f.write(summary.as_text())
    print(summary)

# %%
    X = df[['uhvg','speed','pressure','china_score','index-台风','index-暴雨','index-暴雨预警','index-台风路径']].copy()
    X = sm.add_constant(X)
    y = df['resid']
    model = sm.OLS(y, X, missing='drop').fit()
    resid_r2_direct = model.rsquared
    print(model.summary())
# %%

    def train_model(df_: pl.DataFrame, X_cols: list[str]):
        if len(df_) <= 10:
            return None
        
        trade_days = df_.select(max=pc('date').max(), min=pc('date').min()).select(pc('max') - pc('min')).item()
        assert isinstance(trade_days, datetime.timedelta)
        
        df = df_.to_pandas().copy()
        df['date'] = pd.to_datetime(df['date'])

        X = sm.add_constant(df['uhvg'])
        y = df['fgdm']
        mkt_model = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')
        df['resid'] = mkt_model.resid
        
        
        X = df[list(X_cols)]
        X = sm.add_constant(X)
        # X['const'] = 1 # 偶尔 add_constant 不生效

        y = df['resid']
        resid_model = sm.OLS(y, X, missing='drop').fit()

        return {
            'mkt_model': mkt_model,
            'resid_model': resid_model,
            'info': {
                'sample_days': len(df),
                'sample_days_per_year': len(df) / np.ceil(trade_days.days / 365),
                'mkt_r2': mkt_model.rsquared,
                'resid_r2': resid_model.rsquared,
            }
        }

    to_show_rows = []
    speed_thresholds = np.arange(1.0, 1.950, 0.01)
    # speed_thresholds = np.arange(1.5, 1.76, 0.01)
    # speed_thresholds = np.arange(5, 10, 0.1)

    x_cols = ['uhvg','speed','china_score','index-台风','index-暴雨预警','index-台风路径']
    # x_cols = ['uhvg','speed','china_score']
    # x_cols = ['uhvg','speed']
    # x_cols = ['uhvg']

    for speed_threshold in speed_thresholds:
        df = pct_df.filter(
            # pc('date') >= datetime.date(2024, 5, 1),
            pc('speed') > speed_threshold,
            # pc('index-暴雨') > speed_threshold,
        )
        r = train_model(df, x_cols)
        if r is None:
            continue
        
        to_show_rows.append({
            'speed_thr': speed_threshold,
            'mkt_model_rsquared': r['mkt_model'].rsquared,
            'resid_model_rsquared': r['resid_model'].rsquared,

            **r['info'],
        })
        # print(f'{speed_threshold:.3f}: {len(df)}, {r["info"]["resid_r2"]:.3f}, {r["info"]["mkt_r2"]:.3f}')
    if not to_show_rows:
        continue
    to_show_df = pl.DataFrame(to_show_rows)
    if not to_show_df.is_empty():
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        axes[0].plot(to_show_df['speed_thr'], to_show_df['mkt_model_rsquared'], label='base')
        # axes[0].plot(to_show_df['speed_thr'], to_show_df['mkt_model_adjrsquared'], label='base_adj')
        axes[0].plot(to_show_df['speed_thr'], to_show_df['resid_model_rsquared'], label='resid_with_tdfg')
        axes[0].axhline(0.1, color='green', linestyle='--')

        # axes[0].plot(to_show_df['speed_thr'], to_show_df['resid_model_adjrsquared'], label='resid_with_tdfg_adj')
        axes[1].plot(to_show_df['speed_thr'], to_show_df['sample_days'])
        axes[1].axhline(80, color='red', linestyle='--')
        # axes[1].scatter(to_show_df['speed_thr'], to_show_df['sample_days'])
        axes[1].set_xlabel('speed_thr')
        axes[1].set_ylabel('sample_days')
        
        # plt.plot(speed_thresholds, resid_r2_with_tdfg, label='resid_with_tdfg')
        # plt.plot(speed_thresholds, fgdm_r2_with_tdfg, label='fgdm_with_tdfg')
        axes[0].legend()
        axes[1].legend()
        plt.savefig(curr_stock_dir / f'20-回归.png')
        # plt.close()
        print(model.summary())

# %%
    min_speed_threshold = 1.5
    min_speed_threshold = to_show_df.filter(pc('resid_model_rsquared') > 0.1)[0, 'speed_thr']
    if hasattr(min_speed_threshold, '__len__'):
        # 通常就是空列表, 没有符合条件的 thr
        continue

    print(f'{min_speed_threshold=}')
    min_speed_threshold = max(min_speed_threshold, 1.2)
    print(f'{min_speed_threshold=:.4f}')
    df = pct_df.filter(
        pc('speed') > min_speed_threshold,
    )
    return_pct = 1
    
    trade_dts = []
    return_pcts = []
    trade_records = []
    
    test_dfs = []
    fgdm_pred_totals = []
    for year in range(2018, 2026):
        train_df = df.filter(
            (year - pc('date').dt.year()) > 0,
            (year - pc('date').dt.year()) < 2,
        )
 
        test_df = df.filter(pc('date').dt.year() == year)
        if len(train_df) <= 10:
            print(f"{year}: train_df={len(train_df)}, test_df={len(test_df)}. data is too short")
            continue
        
        r = train_model(train_df, x_cols)
        if r is None: continue

        mkt_model = r['mkt_model']
        resid_model = r['resid_model']

        try:
            # 数量太少的时候, sm.add_constant 不生效, 手工添加效果也不佳

            # 市场模型预测
            fgdm_pred = mkt_model.predict(sm.add_constant(test_df[['uhvg']].to_pandas()))

            # 残差模型预测
            resid_pred = resid_model.predict(sm.add_constant(test_df[x_cols].to_pandas()))
        except Exception as e:
            continue

        # 拼合
        fgdm_true = test_df['fgdm'].to_numpy()
        fgdm_pred_total = fgdm_pred + resid_pred  # 市场+台风残差
        
        test_dfs.append(test_df)
        fgdm_pred_totals.append(fgdm_pred_total)

    if not test_dfs:
        continue

    test_df = pl.concat(test_dfs)
    fgdm_pred_total = np.concatenate(fgdm_pred_totals)
    fgdm_pred_total = pl.from_numpy(fgdm_pred_total, schema=['fgdm_pred_total'])
    assert test_df.shape[0] == fgdm_pred_total.shape[0]
    
    bt_df = pl.concat([
        test_df, 
        fgdm_pred_total,
    ], how='horizontal').with_columns(
        signal = pc('fgdm_pred_total').sign(),
    )
    bt_df = curr_stock_ohlcv_df.join(bt_df, on='date', how='left').fill_null(0.0).sort('date').join(
        tdfg_baidu_index_df,
        on='date',
        how='left',
    ).join(上证指数df, on='date', how='left')
    bt_df
# %%
    FEE = 0.06 / 100
    SLIPPAGE = 0.0 

    class TyphoonEventStrategy(Strategy):
        def init(self):
            self.last_open_dt = self.data.index[0]
            self.I(lambda : self.data.close, name='上证', overlay=False)
            self.I(lambda : self.data.fgdm_pred_total, name='pred', overlay=False)
            for key in ['index-台风', 'index-台风路径', 'index-暴雨', 'index-暴雨预警']:
                self.I(lambda : self.data[f'{key}_right'], name=key, overlay=False)
        def next(self):
            p = 1
            size = 0.5
            
            pred = self.data.fgdm_pred_total[-1]

            if pred > p:
                self.buy(size=size)
                self.last_open_dt = self.data.index[-1]
            # elif pred < -p:
            #     self.position.close()
            else:
                if self.position and (self.data.index[-1] - self.last_open_dt) >= datetime.timedelta(days=0): 
                    self.position.close()

    def run_bt():
        bt = Backtest(
            bt_df.to_pandas().set_index('date'), TyphoonEventStrategy, 
            cash=1_0000, 
            commission=FEE + SLIPPAGE, 
            # exclusive_orders=True, # 开启后第二笔多单也会导致第一笔平仓
            # hedging=True,
            finalize_trades=True
        )
        stats = bt.run()
        return bt, stats

    bt, stats = run_bt()
    stats_dict = dict(stats.items())
    for k in ('_strategy', '_trades', '_equity_curve'):
        stats_dict.pop(k)
    for k, v in stats_dict.items():
        if isinstance(v, pd.Timestamp):
            stats_dict[k] = v.to_pydatetime()
        if isinstance(v, pd.Timedelta):
            stats_dict[k] = f'{v.days}'

    if 'get_ipython' in globals():
        bt.plot()
    stats
# %%
# %%
    info = {
        'name': curr_stock_row['name'],
        'ols-上证指数对风电的解释度': mkt_model.rsquared,
    }
    nice_records = to_show_df.filter(
        pc('resid_model_rsquared') > 0.1
    )
    info['残差R2*样本天数'] = nice_records.select(
        s = pc('resid_model_rsquared') * pc('sample_days_per_year')
    ).sum().item()
    info['递减性'] = to_show_df.select(
        s = pc('mkt_model_rsquared') < pc('mkt_model_rsquared').first()
    ).sum().item() / len(to_show_df)
    info['递减性'] = info['递减性']**2

    info['summary'] = info['残差R2*样本天数'] * info['递减性']

    info['nr_mkt_r2'] = nice_records['mkt_model_rsquared'].mean()
    info['nr_resid_r2'] = nice_records['resid_model_rsquared'].mean()
    info['nr_mean_days'] = nice_records['sample_days'].mean()
    info['nr_mean_days_per_year'] = nice_records['sample_days_per_year'].mean()
    info['nr_len'] = len(nice_records)
    info['nr_len_unique'] = ",".join(nice_records['sample_days'].unique().cast(pl.String))
    
    # 设置stats._equity_curve的index名称为dt
    stats._equity_curve.index.name = 'dt'
    cum_return = pl.from_pandas(stats._equity_curve, include_index=True).group_by(pc('dt').dt.year()).agg(
        pc('Equity').first().alias('year_start_equity'),
        pc('Equity').last().alias('year_end_equity'),
    ).sort('dt').with_columns(
        year_equity_pct = (pc('year_end_equity') - pc('year_start_equity')) / pc('year_start_equity'),
    ).select(
        year='dt',
        cum_return = pc('year_equity_pct').add(1).cum_prod(),
    )
    cum_return_dic = dict(zip(cum_return['year'], cum_return['cum_return']))
    for year in range(2017, 2026):
        ret = cum_return_dic.get(year, 1) - 1
        info[f'r_{year}'] = ret

    info |= stats_dict

    info

# %%
    with open(curr_stock_dir / f'80-info.json', 'w') as f:
        f.write(orjson.dumps(info, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode('utf-8'))
    total_results[curr_stock_row['name']] = info
    
    rows = []
    for name, info in total_results.items():
        rows.append(info)

    year_list = set([i.split('-')[0] for i in next(iter(total_results.values())).keys() if i.startswith('20')])

    res_df = pl.DataFrame(rows).fill_null(0)
    res_df.write_excel(result_dir / '100-total_results.xlsx')

# %%
# result_dir = Path('/root/projects/quant/src/d20250926_台风与风电/data/result-20250930-185854')
total_results = {}
for stock_index, curr_stock_row in tqdm(enumerate(stock_df.iter_rows(named=True)), total=len(stock_df)):
    curr_stock_dir = result_dir / f'{curr_stock_row["name"]}'
    p = curr_stock_dir / f'80-info.json'
    if not p.exists():
        continue

    with open(curr_stock_dir / f'80-info.json', 'rb') as f:
        info = orjson.loads(f.read())
    total_results[curr_stock_row['name']] = info
# %%
with open(result_dir / f'90-total_results.json', 'w') as f:
    f.write(orjson.dumps(total_results, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode('utf-8'))
# %%
rows = []
for name, info in total_results.items():
    rows.append(info)

year_list = set([i.split('-')[0] for i in next(iter(total_results.values())).keys() if i.startswith('20')])

res_df = pl.DataFrame(rows).filter(
    pc('# Trades') != 0
).fill_null(0).join(all_stock_df, on='name')
res_df.write_excel(result_dir / '100-total_results.xlsx')
res_df
# %%
exit(0)
# %%
# tdfg_dates: dict[datetime.datetime, list] = defaultdict(list)
# for row in tdfg_df.rows(named=True):
#     tdfg_name = f'{row['风暴名称']}-{row["year"]}'
#     date_range = []
#     start = row['start_date']
#     while start <= row['end_date']:
#         date_range.append(start)
#         start += datetime.timedelta(days=1)
    
#     for d in date_range:
#         tdfg_dates[d].append(tdfg_name)
#     print(row)
# %%
start_date = datetime.date(2022, 6, 1)
end_date = datetime.date(2022, 10, 30)

r = merged_final_df.sort('date').filter(
    pc('date') > start_date,
    pc('date') < end_date
).select(
    'date',
    pc('norm_close').pct_change().alias('fgdm_pct_change'),
).join(uhvg_pct_df, on='date', how='inner').select(
    'date',
    'fgdm_pct_change',
    uhvg_pct_change = 'return_pct',
).filter(
    pc('fgdm_pct_change').is_null().not_(),
    pc('uhvg_pct_change').is_null().not_(),
).with_columns(
    diff = pc('fgdm_pct_change') - pc('uhvg_pct_change'),
)
r
# %%
px.line(
    r,
    x='date',
    y='diff',
    # color='tdfg_name',
)
# %%
r = pl.from_pandas(ts.pro_bar(ts_code='838670.BJ', start_date='20150519', end_date='20230630', asset='E', )).sort('trade_date').with_columns(date=pl.col('trade_date').str.to_date('%Y%m%d'))
plt.plot(r['date'], r['close'])
# %%

import akshare as ak

r = ak.stock_zh_a_hist(symbol="838670", period="daily", start_date="20150519", end_date='20230630', adjust="").sort_values('日期')
plt.plot(r['日期'], r['开盘'])

plt.plot(r['日期'], r['收盘'])
r