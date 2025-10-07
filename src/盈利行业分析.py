# %%
__import__("os").chdir(__import__("os").path.dirname(__file__))

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
    Iterable,
    Literal,
    Mapping,
    ParamSpec,
    Sequence,
    TypeGuard,
    TypeVar,
    cast,
)
import duckdb
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
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import (
    cache_df,
    fetch_all_stock_list,
    fetch_daily_bar,
    fetch_daily_basic,
    set_cache_dir,
    to_pl_wrapper,
    with_try_wrapper,
)

pc = pl.col
data_dir = Path(__file__).parent / 'data' / 'raw'
set_cache_dir(data_dir)

result_dir = Path(__file__).parent / 'data' / 'result-20250930-210515'
result_dir = Path(__file__).parent / 'data' / 'result-test'
result_dir = Path(__file__).parent / 'data' / 'result-test-total-speed1'

pro = ts.pro_api()
start_date = '2005-01-01'
start_date = '2017-05-01'
end_date = '2025-09-26'

# result_dir = Path(__file__).parent / 'data' / 'result-test-old'
# %%
all_stock_df = fetch_all_stock_list().with_columns(
    list_date = pc('list_date').cast(pl.String).str.strptime(pl.Date, format='%Y%m%d'),
).with_columns(
    live_days = (pl.lit(datetime.date.today()) - pc('list_date')).dt.total_days(),
)
all_stock_df
# # %%
# price_stats = []
# for ts_code in tqdm(all_stock_df['ts_code']):
#     df = fetch_daily_bar(ts_code, start_date, end_date)
#     if len(df) < 365:
#         continue
#     r = df.select(
#         pc('ts_code').first().alias('ts_code'),
#         pc('close').min().alias('min_close'),
#         pc('close').max().alias('max_close'),
#         pc('close').median().alias('median_close'),
#     ).to_dicts()[0]
#     price_stats.append(r)
# %%

daily_bar_db = duckdb.connect(data_dir / 'tushare_daily_bar.duckdb')
price_df = daily_bar_db.sql("""
    select 
        first(ts_code) as ts_code, 
        min(close) as min_close, 
        max(close) as max_close, 
        median(close) as median_close, 
        min(date), 
        max(date),
    from tushare_daily_bar
    where date >= ? and date <= ?
    group by ts_code
                 
    order by ts_code
""", params=[start_date, end_date]).pl()
price_df
# %%

return_df = pl.read_excel(result_dir / '90.所有股票指标汇总.xlsx')
return_df = all_stock_df.join(return_df, left_on='name', right_on='name', how='inner').with_columns(
    Duration = pc('Duration').str.replace('d', '').cast(pl.Int32),
    jnyiso = pc('ts_code').str.split('.').list.get(1),
).filter(
    pc('Duration') >= 365,
    pc('name').is_in(['东方精工']).not_(),
    # pc('Duration') >= 365*2,
    # pc('ts_code').str.ends_with('.BJ').not_()
).join(
    price_df, left_on='ts_code', right_on='ts_code', how='left'
)
assert len(return_df.filter(pc('min_close').is_null())) == 0

plt.hist(return_df['CAGR [%]'], bins=100)
plt.title(f'CAGR Distribution. Mean={return_df['CAGR [%]'].mean():.2f}%, Std={return_df['CAGR [%]'].std():.2f}%')
plt.xlabel('CAGR [%]')
plt.axvline(0, color='red', linestyle='--')
return_df
# %%
fig, axes = plt.subplots(3, figsize=(5, 12))
for i, col in enumerate(['min_close', 'median_close', 'max_close']):
    axes[i].scatter(np.log10(return_df[col]), return_df['CAGR [%]'], s=2, alpha=0.5)
    axes[i].set_title(f'{col} vs CAGR')
    axes[i].set_xlim(0, 5)
fig.savefig('price_vs_cagr.png')

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
axes[0].scatter(return_df['live_days'] / 365, return_df['CAGR [%]'], s=2)
axes[0].axhline(0, color='gray', linestyle='--')
axes[0].set_title('Live Years vs CAGR')
axes[0].set_xlabel('Live Years')
axes[0].set_ylabel('CAGR [%]')

grouped_year = 3
grouped_by_live_year = return_df.group_by((pc('live_days') / 365 / grouped_year).cast(pl.Int32).alias('live_year')).agg(
    pc('CAGR [%]').mean().alias('CAGR_mean'),
    pc('CAGR [%]').std().alias('CAGR_std'),
    pl.len().alias('count'),
).sort('live_year')
axes[1].bar(grouped_by_live_year['live_year'], grouped_by_live_year['CAGR_mean'], yerr=grouped_by_live_year['CAGR_std'], capsize=5)
axes[1].set_title('Live Years vs CAGR Mean±Std')
axes[1].set_xlabel(f'Live Years ({grouped_year} years per unit)')
axes[1].set_ylabel('CAGR Mean [%]')
axes[1].set_xticks(grouped_by_live_year['live_year'], labels=[f'{x*grouped_year}-{(x+1)*grouped_year}' for x in grouped_by_live_year['live_year']])
axes[1].axhline(0, color='gray', linestyle='--')
plt.tight_layout()
fig.savefig('live_year_vs_cagr.png')
# %%
return_by_jnyiso = return_df.group_by('jnyiso').agg(
    pc('CAGR [%]').mean().alias('CAGR_mean'),
    pc('CAGR [%]').std().alias('CAGR_std'),
    pc('Duration').mean().alias('Duration_mean'),
    pl.len().alias('count'),
).sort('CAGR_mean', descending=True)
return_by_jnyiso
# %%
return_df.select(
    (pc('CAGR [%]') > 0).sum().alias('CAGR > 0'),
    (pc('CAGR [%]') < 0).sum().alias('CAGR < 0'),
    (pc('CAGR [%]') == 0).sum().alias('CAGR = 0'),
)
# %%
with pl.Config(tbl_rows=1000):
    print(return_df['industry'].value_counts(sort=True))
# %%
buy_hold_ret = (pc('Buy & Hold Return [%]') * pc('Exposure Time [%]') / 100)

return_grouped_by_industry = return_df.with_columns(
    ret_diff_ann = (pc('Return [%]') + buy_hold_ret) / pc('Duration') * 365,
).group_by('industry').agg(
    pc('Return (Ann.) [%]').mean().alias('ret_ann'),
    pc('Return (Ann.) [%]').median().alias('ret_ann_med'),
    pc('Return (Ann.) [%]').std().alias('ret_ann_std'),
    buy_hold_ret.mean().alias('buy_hold_ret'),
    # pc('ret_diff_ann').mean().alias('ret_diff_ann'),
    pc('Avg. Drawdown [%]').mean().alias('avg_drawdown'),
    # pc('Sharpe Ratio').mean().alias('sharpe_ratio'),
    pl.len().alias('stock_count'),
    
).with_columns(
    
).sort('ret_ann_med', descending=True).filter(
    pc('stock_count') >= 5,
).drop_nulls()
with pl.Config(tbl_rows=1000, tbl_cols=1000):
    print(return_grouped_by_industry)
return_grouped_by_industry
# %%
split_rows = 4
cols_per_row = (len(return_grouped_by_industry) + split_rows - 1) // split_rows
r = return_grouped_by_industry.sort('ret_ann_med', descending=True)
chunks = [r[i*cols_per_row:(i+1)*cols_per_row] for i in range(split_rows)]

fig, axes = plt.subplots(split_rows, 1, figsize=(3 * split_rows, 10), sharey=True)
for i, chunk in enumerate(chunks):
    axes[i].bar(
        chunk['industry'], chunk['ret_ann_med'], yerr=chunk['ret_ann_std'], 
        capsize=5
    )
    axes[i].set_xticklabels(chunk['industry'], rotation=20)
    axes[i].set_title(f'行业年化收益率中位数 (分组 {i+1}/{split_rows})')
    axes[i].axhline(0, color='gray', linestyle='--')
plt.tight_layout()
fig.savefig('industry_ret_ann_med.png')
# %%
return_grouped_by_province = return_df.with_columns(
    ret_diff_ann = (pc('Return [%]') + buy_hold_ret) / pc('Duration') * 365,
).group_by('area').agg(
    pc('Return (Ann.) [%]').mean().alias('ret_ann'),
    pc('Return (Ann.) [%]').median().alias('ret_ann_med'),
    pc('Return (Ann.) [%]').std().alias('ret_ann_std'),
    buy_hold_ret.mean().alias('buy_hold_ret'),
    pc('ret_diff_ann').mean().alias('ret_diff_ann'), 
    pc('Avg. Drawdown [%]').mean().alias('avg_drawdown'),
    # pc('Sharpe Ratio').mean().alias('sharpe_ratio'),
    pl.len().alias('stock_count'),
).with_columns(

).sort('ret_ann_med', descending=True).filter(
    pc('stock_count') >= 5,
).drop_nulls()
with pl.Config(tbl_rows=1000, tbl_cols=1000):
    print(return_grouped_by_province)
# %%
split_rows = 3
cols_per_row = (len(return_grouped_by_province) + split_rows - 1) // split_rows
r = return_grouped_by_province.sort('ret_ann_med', descending=True)
chunks = [r[i*cols_per_row:(i+1)*cols_per_row] for i in range(split_rows)]

fig, axes = plt.subplots(split_rows, 1, figsize=(3 * split_rows, 10), sharey=True)
for i, chunk in enumerate(chunks):
    axes[i].bar(
        chunk['area'], chunk['ret_ann_med'], yerr=chunk['ret_ann_std'], 
        capsize=5
    )
    axes[i].set_xticklabels(chunk['area'], rotation=20)
    axes[i].set_title(f'省份年化收益率中位数 (分组 {i+1}/{split_rows})')
    axes[i].axhline(0, color='gray', linestyle='--')
plt.tight_layout()
fig.savefig('province_ret_ann_med.png')
# %%
plt.hist(return_grouped_by_industry['ret_diff_ann'], bins=100)
# %%
df = return_df.to_pandas()
# %%
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, plot_importance, plot_tree

# 假设已有 df(DataFrame)，包含这些列：
# 上证-风电:r2, 上证台风-风电:r2, 残差r2*样本天数, dt, train_fgdm_r2, train_resid_r2, CAGR [%]

# %%
rename_map = {
    '上证-风电:r2': 'r2_mkt_wind',
    '上证台风-风电:r2': 'r2_mkttyphoon_wind',

    'train_fgdm_r2': 'train_fgdm_r2',
    'train_resid_r2': 'train_resid_r2',
    'CAGR [%]': 'cagr',
}
df_ = return_df.to_pandas().rename(columns=rename_map)

# %%
feature_cols = ['r2_mkt_wind', 'r2_mkttyphoon_wind', 'train_fgdm_r2', 'train_resid_r2']
X = df_[feature_cols]
y = df_['cagr']

# %%
lgb = LGBMRegressor(n_estimators=500)
lgb.fit(X, y)


def plot_pred(cagr: Iterable, cagr_pred: Iterable):
    cagr = np.array(cagr)
    cagr_pred = np.array(cagr_pred)

    sig_acc = (np.sign(cagr) == np.sign(cagr_pred)).sum() / len(cagr)

    plt.scatter(cagr, cagr_pred, s=2)
    # plt.plot([-15, 15], [-15, 15], 'r--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('实际 CAGR [%]')
    plt.ylabel('预测 CAGR [%]')
    plt.title('CAGR 预测 vs 实际 sig_acc={:.2%}'.format(sig_acc))

plot_pred(y, lgb.predict(X))
# %%
plot_tree(lgb, figsize=(20, 10))
# %%

# %%
feature_cols = ['r2_mkt_wind', 'r2_mkttyphoon_wind',
                'train_fgdm_r2', 'train_resid_r2']
X = df_[feature_cols]
# y = (df_['cagr'] > 0) * 1.0
y = df_['cagr']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5
)

# %% 训练
# lgb = LGBMClassifier()
lgb = LGBMRegressor()
lgb.fit(X_train, y_train)

y_pred = lgb.predict(X_test)
plot_pred(y_test, y_pred)
# %%
cm = confusion_matrix(y_test > 0, y_pred > 0, labels=[0,1])

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, values_format='d')
# %%
plot_tree(lgb, figsize=(20, 10))
# %%
X_const = sm.add_constant(X_train)

# %%
model = sm.OLS(y_train, X_const)
results = model.fit()
y_pred = results.predict(sm.add_constant(X_test))
print(results.summary())
# %%
plot_pred(y_test, y_pred)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df_[['r2_mkt_wind', 'r2_mkttyphoon_wind', 'resid_r2_days',
                  'train_fgdm_r2', 'train_resid_r2', 'cagr']])
# %%
# %%

poly = PolynomialFeatures(degree=3, include_bias=True)
X_poly = poly.fit_transform(X_train)

linreg = LinearRegression()
linreg.fit(X_poly, y_train)

X_poly_test = poly.transform(X_test)
y_pred = linreg.predict(X_poly_test)

plot_pred(y_test, y_pred)
# %%
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.svm import SVR

kr = KernelRidge(kernel="rbf", alpha=1.0, gamma=0.1)
kr.fit(X_train, y_train)
print("Kernel Ridge R2:", r2_score(y_train, kr.predict(X_train)))
plot_pred(y_train, kr.predict(X_train))
# plot_pred(y_test, kr.predict(X_test))