# %%
__import__("os").chdir(__import__("os").path.dirname(__file__))

import concurrent.futures
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict, cast, overload

import duckdb
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import tushare as ts
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


from utils import (
    cache_df,
    fetch_all_stock_list,
    fetch_daily_bar,
    fetch_daily_basic,
    set_cache_dir,
    to_pl_wrapper,
    with_try_wrapper,
)


def unwrap[T](v: T | None) -> T:
    if v is None:
        raise ValueError('unwrap None')
    return v

pc = pl.col
pro = ts.pro_api()
start_date = '2005-01-01'
trade_start_date = '2017-01-01'
end_date = '2025-09-26'

data_dir = Path(__file__).parent / 'data' / 'raw'
set_cache_dir(data_dir)

all_stock_list_df = fetch_all_stock_list()
all_stock_list_df
# %%
# %%

#################################
# 数据准备
#################################


# @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
# def retry_get(ts_code: str, start_date: str, end_date: str, proxies:dict[str, str]={}):
#     return fetch_daily_basic(ts_code, start_date, end_date)

# for ts_code in (pbar := tqdm(fetch_all_stock_list()['ts_code'])):
#     pbar.set_description(ts_code)
#     retry_get(ts_code, start_date, end_date)

# @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
# def retry_get(ts_code: str, start_date: str, end_date: str, proxies:dict[str, str]={}):
#     try:
#         return fetch_daily_bar(ts_code, start_date, end_date)
#     except TypeError:
#         pass
#     except Exception:
#         raise

# for ts_code in (pbar := tqdm(fetch_all_stock_list()['ts_code'])):
#     pbar.set_description(ts_code)
#     retry_get(ts_code, start_date, end_date)

# daily_basic_db = duckdb.connect(data_dir / 'tushare_daily_basic.duckdb')
# # daily_basic_db.sql(f"create table tushare_daily_basic as (select * from '{data_dir}/basic/*.parquet' order by date)")
# daily_bar_db = duckdb.connect(data_dir / 'tushare_daily_bar.duckdb')
# # daily_bar_db.sql(f"CREATE OR REPLACE TABLE tushare_daily_bar as (select * from '{data_dir}/adj-hfq/*.parquet' order by date, change)")

# %%
db_basic = data_dir / "tushare_daily_basic.duckdb"
db_bar   = data_dir / "tushare_daily_bar.duckdb"

con = duckdb.connect(database=":memory:")
con.execute(f"ATTACH '{db_basic}' AS dbb;")
con.execute(f"ATTACH '{db_bar}'   AS dbr;")

con.execute("""
CREATE OR REPLACE VIEW daily_basic AS
SELECT * FROM dbb.tushare_daily_basic;
CREATE OR REPLACE VIEW daily_bar AS
SELECT * FROM dbr.tushare_daily_bar;
""")
# %%
print(con.sql('SELECT * from daily_basic LIMIT 5').df())
'''
     ts_code trade_date  close  turnover_rate  turnover_rate_f  volume_ratio  \
0  000007.SZ   20050104   3.42         0.3639           0.3639          0.74   
1  600061.SH   20050104   3.06         0.2175           0.2175          1.23   
2  002007.SZ   20050104  17.10         0.6493           0.6493          1.06   
3  000729.SZ   20050104  11.46         0.0206           0.0206          0.22   
4  000859.SZ   20050104   2.90         0.2571           0.2571          1.28   

         pe    pe_ttm      pb      ps  ps_ttm  dv_ratio  dv_ttm  total_share  \
0   32.8087       NaN  2.5371  1.4682  1.9495    0.0000     NaN   14359.3664   
1  266.2871  191.2768  2.3459  1.3714  1.2408    0.0000     NaN   37311.5600   
2   32.2821   32.2821  2.3448  4.5302  4.5302    0.0000     NaN    6700.0000   
3   31.7981   30.1982  1.8585  2.2443  1.7486    5.4871  5.4871   67127.8189   
4  246.0265       NaN  1.2249  1.4081  1.2808    0.0000     NaN   42048.0000   

   float_share  free_share     total_mv      circ_mv       date  
0    7956.0961   7956.0961   49109.0331   27209.8487 2005-01-04  
1   12168.0000  12168.0000  114173.3736   37234.0800 2005-01-04  
2    2200.0000   2200.0000  114570.0000   37620.0000 2005-01-04  
3   19105.3689  19105.3689  769284.8046  218947.5276 2005-01-04  
4   18720.0000  18720.0000  121939.2000   54288.0000 2005-01-04  


ts_code 	str 	TS股票代码
trade_date 	str 	交易日期
close 	float 	当日收盘价
turnover_rate 	float 	换手率（%）
turnover_rate_f 	float 	换手率（自由流通股）
volume_ratio 	float 	量比
pe 	float 	市盈率（总市值/净利润， 亏损的PE为空）
pe_ttm 	float 	市盈率（TTM，亏损的PE为空）
pb 	float 	市净率（总市值/净资产）
ps 	float 	市销率
ps_ttm 	float 	市销率（TTM）
dv_ratio 	float 	股息率 （%）
dv_ttm 	float 	股息率（TTM）（%）
total_share 	float 	总股本 （万股）
float_share 	float 	流通股本 （万股）
free_share 	float 	自由流通股本 （万）
total_mv 	float 	总市值 （万元）
circ_mv 	float 	流通市值（万元）
'''
# %%
print(con.sql('SELECT * from daily_bar LIMIT 5').df())
'''
     ts_code trade_date       open       high        low      close  \
0  600652.SH   20050104  19105.920  19215.720  18593.500  18666.700   
1  600651.SH   20050104  29324.080  29324.080  28326.220  28809.060   
2  600653.SH   20050104  22399.970  22483.860  22064.390  22148.280   
3  000001.SH   20050104   1260.782   1260.782   1238.179   1242.774   
4  600616.SH   20050104   1135.450   1136.770   1106.300   1116.900   

   pre_close   change  pct_chg         vol       amount       date  
0  19252.330 -585.630   -3.040    12442.90     6424.199 2005-01-04  
1  29324.080 -515.020   -1.760     8590.88     7665.145 2005-01-04  
2  22483.860 -335.580   -1.490    21101.85     5592.564 2005-01-04  
3   1266.496  -23.722   -1.873  8161770.00  4418452.072 2005-01-04  
4   1136.770  -19.870   -1.750     1432.00     1206.781 2005-01-04  
'''
# %%
mkt_df = con.sql(f"""
SELECT 
    b.ts_code,
    b.date,
    DATE(b.date + INTERVAL '1' DAY) AS EntryTime,

    AVG(b.amount) OVER (
        PARTITION BY b.ts_code
        ORDER BY b.date
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) AS adv20,
    AVG(b.pct_chg) OVER (
        PARTITION BY b.ts_code
        ORDER BY b.date
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS adv5_pct_chg,
    STDDEV(b.pct_chg) OVER (
        PARTITION BY b.ts_code
        ORDER BY b.date
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS std5_pct_chg,

    b.open, b.high, b.low, b.close, b.pre_close, b.vol, b.amount, b.pct_chg,

    (b.close / LAG(b.close, 5) OVER (PARTITION BY b.ts_code ORDER BY b.date) - 1) AS ret_5,
    (b.open = b.high AND b.high = b.low AND b.low = b.close AND b.pct_chg > 0) AS is_up_limit,
    (b.open = b.high AND b.high = b.low AND b.low = b.close AND b.pct_chg < 0) AS is_down_limit,
    basic.*
FROM daily_bar AS b
JOIN daily_basic AS basic
  ON b.ts_code = basic.ts_code and b.date = basic.date
WHERE b.date <= '{end_date}'
  AND b.date >= DATE('{trade_start_date}') - INTERVAL '10' DAY

""").pl()
mkt_df.write_parquet(data_dir / 'mkt_df.parquet')
mkt_df
# %%
mkt_df.filter(pc('is_up_limit')).sort('pct_chg', descending=True)
# %%
trades = []
result_dir = Path('data/result-test-total-speed1/')
cache_p = result_dir / 'trades_df.parquet'
if cache_p.exists():
    trades_df = pl.read_parquet(cache_p)
    print(f'从缓存读取 {cache_p} {trades_df.height}行')
else:
    files = list(result_dir.rglob('70.回测-交易明细.xlsx'))
    print(f'找到 {len(files)} 个交易明细文件')

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(pl.read_excel, p, schema_overrides={
            'SL': pl.Float64,
            'TP': pl.Float64,
            'Tag': pl.String,
            'Exit_pred': pl.Float64,
            'Exit_speed': pl.Float64,
        }): p for p in files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            p = futures[future]
            try:
                df = future.result()
                df = df.with_columns(
                    name = pl.lit(p.parent.name),
                )
                trades.append(df)
            except Exception as e:
                print(f'文件 {p} 读取失败: {e}, 文件为空？')
    trades_df: pl.DataFrame = pl.concat(trades).select(
        pl.all().exclude('SL', 'TP', 'Tag'),
    ).join(all_stock_list_df[['name', 'ts_code']], on='name')
    trades_df.write_parquet(result_dir / 'trades_df.parquet')
    del trades

trades_df
# %%
trades_df.group_by('name').agg(
    pc('PnL').sum(),
    pc('Commission').sum(),
    pc('ReturnPct').sum(),
).sort('PnL')
# %%
def stat_df(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        count = pc('ReturnPct').count(),
        ret_pct_mean = pc('ReturnPct').mean() * 100,
        ret_pct_std  = pc('ReturnPct').std() * 100,
        ret_pct_p5 = pc('ReturnPct').quantile(0.05) * 100,
        ret_pct_p25 = pc('ReturnPct').quantile(0.25) * 100,
        ret_pct_p50 = pc('ReturnPct').quantile(0.5) * 100,
        ret_pct_p75 = pc('ReturnPct').quantile(0.75) * 100,
        ret_pct_p95 = pc('ReturnPct').quantile(0.95) * 100,
        win_rate = (pc('ReturnPct') > 0).mean() * 100,
        skewness = pc('ReturnPct').skew(),
    )

stat_df(trades_df)
# %%
stat_df(trades_df.filter(
    pc('EntryPrice') < 50,
    pc('EntryPrice') > 5,
    
    pc('name').str.contains('ST').not_()
))

# %%

def not_limit[T: pl.DataFrame | pl.LazyFrame](trades_df: T, mkt_df: T) -> T:
    r = trades_df.join(
        mkt_df[['ts_code', 'EntryTime', 'is_up_limit', 'is_down_limit']], # type: ignore
        on=['ts_code', 'EntryTime'],
    ).filter(
        pc('is_up_limit').not_(),
    ).select(
        pl.all().exclude(['is_up_limit', 'is_down_limit']),
    ).join(
        mkt_df[['ts_code', 'EntryTime', 'is_up_limit', 'is_down_limit']], # type: ignore
        left_on=['ts_code', 'ExitTime'], right_on=['ts_code', 'EntryTime'],
    ).filter(
        pc('is_down_limit').not_(),
    ).select(
        pl.all().exclude(['is_up_limit', 'is_down_limit']),
    )
    return cast(T, r)

trades_df = pl.read_parquet(cache_p)
print(trades_df.shape)
trades_df = not_limit(trades_df, mkt_df)
print(trades_df.shape)
stat_df(trades_df.filter(
    pc('EntryPrice') < 50,
    # pc('EntryPrice') > 5,
    
    pc('name').str.contains('ST').not_()
))


# %%
draw_df = trades_df.filter(
    pc('EntryPrice') < 20,
    pc('EntryPrice') > 5,
    
    pc('name').str.contains('ST').not_()
)
plt.scatter(
    draw_df['EntryPrice'], draw_df['ReturnPct'] * 100,
    alpha=0.1,
    s=1,
)
# %%
# 按 -ret_5 取 top k
top_k = 50


for field_ in ['turnover_rate', 'pe', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_mv', 'float_share', 'free_share', 'total_mv', 'circ_mv']:
# for field in ['turnover_rate']:

    for sig in [1, -1]:
        top_k_col = pc(field_) * sig
        print(f'按 {field_} {"降" if sig==1 else "升"}序取 topk')

        res = []
        rdfs: dict[int, pl.DataFrame] = {}

        for top_k in range(1, 11):
            r = trades_df.lazy().join(
                mkt_df.lazy(),
                on=['ts_code', 'EntryTime'],
            ).filter(
                # pc('pct_chg') >= 0,
                pc('close') > 5,
                # pc('close') < 20,
            # ).with_columns(
            #     rk_turnover_rate = pc('turnover_rate').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
            # ).filter(
            #     pc('rk_turnover_rate') > 0.2,
            #     pc('rk_turnover_rate') < 0.8,
            ).group_by('EntryTime').agg(
                pl.len().alias('num_trades'),
                pl.all().top_k_by(top_k_col, top_k)
            ).select(
                'EntryTime',
                'name',
                pc('ReturnPct').alias('ReturnPctList'),
                'num_trades',
                pc('ret_5'),
                pc('ReturnPct').list.mean(),
            ).sort('EntryTime').collect()
            # print(f'[{top_k}]', r['ReturnPct'].sum(), stat_df(r).to_dicts()[0])
            res.append({
                'top_k': top_k,
                **stat_df(r).to_dicts()[0],
            })
            rdfs[top_k] = r

        with pl.Config(tbl_cols=100, tbl_rows=21, float_precision=2):
            print(pl.DataFrame(res).sort('top_k'))
# %%

res = []
rdfs: dict[int, pl.DataFrame] = {}

key = pc('turnover_rate')
r = trades_df.join(
    mkt_df,
    on=['ts_code', 'EntryTime'],
).with_columns(
    rk = key.rank(descending=True).over('EntryTime') / pl.count().over('EntryTime')
)
r
# %%
plt.hist(r['rk'], bins=100)
# %%
mkt_df
# %%
key = pc('turnover_rate')

res = []
rdfs: dict[int, pl.DataFrame] = {}

for top_k in range(1, 11, 1):
    base_p = 0.5

    r = trades_df.lazy().join(
        mkt_df.lazy(),
        on=['ts_code', 'EntryTime'],
    ).filter(
        # pc('pct_chg').over('ts_code', 'EntryTime').mean() > 0,
        pc('close') > 5,
        # pc('close') < 100,
        # pc('ret_5') < 0.0,
        # pc('close') < 20,

    ).with_columns(
        rk_turnover_rate = pc('turnover_rate').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
        rk_adv20 = pc('adv20').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
        rk_ret_5 = pc('ret_5').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
        rk_ps = pc('ps').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
        rk_total_mv = pc('total_mv').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
        rk_amount = pc('amount').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
        rk_std5_pct_chg = pc('std5_pct_chg').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
    ).filter(
        pc('rk_turnover_rate') > base_p,
        pc('rk_turnover_rate') < 0.99,

        pc('rk_adv20') > base_p,
        pc('rk_adv20') < 0.99,

        pc('std5_pct_chg') > 1,
        # pc('rk_std5_pct_chg') > 0.1,

        # pc('ret_5').abs() < 0.2,
        # pc('rk_ret_5') < 0.5,
        # pc('rk_ret_5') > 0.01,
        # (pc('rk_ret_5') > 0.2) & (pc('rk_ret_5') < 0.8),

        pc('rk_total_mv') > base_p,
        pc('rk_total_mv') < 0.99,

        pc('rk_ps') > base_p,
        # pc('rk_ps') < 0.99,

        # pc('rk_amount') > 0.8,
        # pc('ps') < 5,
    ).group_by('EntryTime').agg(
        pl.len().alias('num_trades'),
        # pl.all(),
        # pl.all().top_k_by(-pc('ps'), top_k), # 0.43+-4.39 53.68% 0.29, k越大越坏
        # pl.all().top_k_by(-pc('total_mv'), top_k), # 0.57 +-4.30 52.45% 1.19
        pl.all().top_k_by(-pc('total_mv') * pc('ps'), top_k), # 0.57 +-4.30 52.45% 1.19
        
        
    ).select(
        'EntryTime',
        'name',
        'ts_code',
        'ps',
        pc('ReturnPct').alias('ReturnPctList'),
        'num_trades',
        pc('ret_5'),
        pc('total_mv'),
        pc('ReturnPct').list.mean(),
    ).sort('EntryTime').collect()
    # print(f'[{top_k}]', r['ReturnPct'].sum(), stat_df(r).to_dicts()[0])
    res.append({
        'top_k': top_k,
        **stat_df(r).to_dicts()[0],
    })
    rdfs[top_k] = r

with pl.Config(tbl_cols=100, tbl_rows=21, float_precision=2):
    print(pl.DataFrame(res).sort('top_k'))
# %%
r = rdfs[6].select(pl.all())

for row in r.tail(30).iter_rows(named=True):
    s = f'[{row["EntryTime"]}] {row["ReturnPct"]: 1.2%}| in{row["num_trades"]:> 4d} | '
    for name, ret in zip(row['name'], row['ReturnPctList']):
        s += f'{name}\t{ret: 1.2%}\t'

    print(s)
# %%
return_pcts = r.select(
    'EntryTime',
    ret = (1 + pc('ReturnPct')).cum_prod()
)
plt.plot(return_pcts['EntryTime'], return_pcts['ret'])
# %%
r['name'].list.explode().unique().sort()
# %%

sns.histplot(r['ReturnPctList'].list.explode() * 100, kde=True, stat='density', bins=100)
plt.axvline(0, color='red', linestyle='--', linewidth=1)
# %%
sns.violinplot(
    r.select(
        pc('EntryTime').dt.month().alias('month'),
        pc('ReturnPct') * 100,
    ).sort('month').to_pandas(),
    x='month',
    y='ReturnPct',
    density_norm='count',
    inner='quartile',
)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
# %%
r = trades_df.lazy().join(
    mkt_df.lazy(),
    on=['ts_code', 'EntryTime'],
).filter(
    # pc('pct_chg') >= 0,
    # pc('close') > 5,
    pc('close') < 100,
    # pc('ret_5').abs() > 0.05,
    # pc('close') < 20,

).with_columns(
    rk_turnover_rate = pc('turnover_rate').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
    rk_adv20 = pc('adv20').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
    rk_ret_5 = pc('ret_5').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
    rk_ps = pc('ps').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
    rk_total_mv = pc('total_mv').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
    rk_amount = pc('amount').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
    rk_std5_pct_chg = pc('std5_pct_chg').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
    rk_avg5_pct_chg = pc('adv5_pct_chg').rank(descending=True).over('EntryTime') / pl.len().over('EntryTime'),
).sort('EntryTime').collect()
r
# %%
draw_cols = ['rk_turnover_rate', 'rk_adv20', 'rk_ret_5', 'rk_ps', 'rk_total_mv', 'std5_pct_chg', 'adv5_pct_chg']
fig, axes = plt.subplots(1, 7, figsize=(21, 4), sharey=True)
for draw_col, ax in zip(draw_cols, axes):
    ax.scatter(r[draw_col], r['ReturnPct'], alpha=0.1, s=1)
    ax.set_title(draw_col)
# plt.tight_layout()
# %%
draw_cols = ['rk_turnover_rate', 'rk_adv20']

for draw_col in draw_cols:
    plt.scatter(r[draw_col], r['ReturnPct'], alpha=0.1, s=1)
plt.legend()
# %%
trade_list_df = rdfs[3]
# trade_list_df.write_parquet('data/trade_list_df.parquet')
# trade_list_df
# mkt_df.write_parquet('data/mkt_df.parquet')
# mkt_df
# %%
all_ts_codes = trade_list_df['ts_code'].explode().unique()
all_dates = trade_list_df['EntryTime'].unique()
print(f'共 {len(all_ts_codes)} 只股票, {len(all_dates)} 个交易日')
date_to_mkt = {
    cast(date, date_): df.lazy()
    for (date_, ), df in mkt_df.filter(
        pc('ts_code').is_in(all_ts_codes),
        pc('EntryTime').is_in(all_dates),
    ).group_by('date')
}
class MktFinder:
    def __init__(self, mkt_df: pl.DataFrame, all_ts_codes: list[str] | None = None) -> None:
        self.mkt_df = mkt_df
        self.all_ts_codes = all_ts_codes

        # @lru_cache(maxsize=10)
        # def cache(date_: date) -> pl.DataFrame | None:
        #     df = self.mkt_df.filter(pc('date') == date_)
        #     if self.all_ts_codes is not None:
        #         df = df.filter(pc('ts_code').is_in(self.all_ts_codes))
        #     df = df.collect()
        #     if not len(df):
        #         return None
        #     return df
        # self.cache = cache
        self._cache = {}

    def cache(self, date_: date) -> pl.DataFrame:
        if date_ in self._cache:
            return self._cache[date_]
        self._cache = {
            date_: df 
            for (date_, ), df in self.mkt_df.filter(
                pc('date') >= date_,
                pc('date') < date_ + timedelta(days=10),
            ).group_by('date')
        }
        return self._cache[date_]

    @overload
    def find_mkt(self, date_: date, ts_code: str, cols: Literal['open']) -> float | None: ...
    @overload
    def find_mkt(self, date_: date, ts_code: str, cols: Literal['close']) -> float | None: ...

    @overload
    def find_mkt(self, date_: date, ts_code: str, cols: str) -> Any | None: ...

    @overload
    def find_mkt(self, date_: date, ts_code: str, cols: list[str]) -> dict[str, Any] | None: ...

    def find_mkt(self, date_: date, ts_code: str, cols: list[str] | str) -> dict[str, Any] | Any | None:
        df = self.cache(date_)
        if isinstance(cols, str):
            selected_cols = [cols]
        else:
            selected_cols = cols

        df = df.lazy().filter(pc('ts_code') == ts_code).select(*selected_cols).collect()
        if not len(df):
            return None
        r = df.to_dicts()[0]
        if isinstance(cols, str):
            return r[cols]
        return r

    # def has_day(self, date_: date) -> bool:
    #     return date_ in self.date_to_mkt

mkt_finder = MktFinder(mkt_df)
# mkt_finder.has_day(date(2024, 10, 24))
mkt_finder.find_mkt(date(2024, 10, 24), '000906.SZ', ['open', 'close'])

# %%

@dataclass(frozen=True)
class UnmatchedOrder:
    ts_code: str
    name: str

    entry_time: date
    entry_size: int

    belong_trade: 'UnclosedTrade | None'

    def fill(self, price: float, fee: float) -> 'Order':
        return Order(
            ts_code=self.ts_code,
            name = self.name,
            entry_size=self.entry_size,
            entry_time=self.entry_time,
            entry_price=price,
            belong_trade=self.belong_trade,
            commission=price * self.entry_size * fee, 
        )

@dataclass(frozen=True)
class Order(UnmatchedOrder):
    entry_price: float
    commission: float

    @property
    def amount(self):
        return self.entry_size * self.entry_price

    @property
    def buy_amount(self):
        return self.amount + self.commission

    @property
    def sell_amount(self):
        return self.amount - self.commission

@dataclass(frozen=True)
class UnclosedTrade:
    open_order: Order

    @property
    def ts_code(self) -> str:
        return self.open_order.ts_code
    @property
    def name(self) -> str:
        return self.open_order.name
    @property
    def EntryTime(self) -> date:
        return self.open_order.entry_time

    def close(self, close_order: Order) -> 'Trade':
        return Trade(
            open_order=self.open_order,
            close_order=close_order,
        )

    def calc_pnl(self, close_price: float, fee: float) -> float:
        mock_close_order = self.open_order.fill(close_price, fee)
        return mock_close_order.sell_amount - self.open_order.buy_amount

    def calc_return_pct(self, close_price: float, fee: float) -> float:
        pnl = self.calc_pnl(close_price, fee)
        return pnl / self.open_order.amount

@dataclass(frozen=True)
class Trade(UnclosedTrade):
    close_order: Order

    @property
    def ExitTime(self) -> date:
        return self.close_order.entry_time
    @property
    def EntryPrice(self) -> float:
        return self.open_order.entry_price
    @property
    def pnl(self) -> float:
        return self.close_order.sell_amount - self.open_order.buy_amount
    @property
    def ReturnPct(self) -> float:
        return self.pnl / self.open_order.amount

    def __repr__(self) -> str:
        return f'Trade({self.EntryTime}~{self.ExitTime}, {self.ts_code} {self.name}, ret={self.ReturnPct:1.2%}, pnl={self.pnl:1.0f})'

class TradeList(TypedDict):
    '''{'EntryTime': datetime.date(2018, 3, 27),
 'name': ['浙商中拓', '创新新材', '风神股份'],
 'ts_code': ['000906.SZ', '600361.SH', '600469.SH'],
 'ps': [0.1193, 0.2591, 0.367],
 'ReturnPctList': [-0.01006813271604945,
  0.1245283762057879,
  0.013099841017488],
 'num_trades': 202,
 'ret_5': [-0.02135774218154074, -0.04155339805825242, -0.0807635829662261],
 'total_mv': [393325.6011, 320253.6086, 268833.5201],
 'ReturnPct': 0.042520028169075484}
    '''
    EntryTime: date
    name: list[str]
    ts_code: list[str]
    ps: list[float]
    ReturnPctList: list[float]
    num_trades: int
    ret_5: list[float]
    total_mv: list[float]
    ReturnPct: float


cash_init = 1_0000_0000
cash = cash_init
fee = 6 / 10000  # 万6
single_day_open_size = 1
stop_loss_p = -5 / 100

unmatched_orders: list[UnmatchedOrder] = []
unclosed_trades: list[UnclosedTrade] = []
finished_trades: list[Trade] = []

all_trade_days: list[date] = mkt_df['date'].unique().sort().to_list()
all_will_trade_list = {
    d['EntryTime']: TradeList(**d)
    for d in trade_list_df.to_dicts()
}
trade_idx = 0

cash_curve = []
for curr_day in tqdm(all_trade_days):
    # print(curr_day, f'{unmatched_orders=}')

    # 先处理上轮刚提交未撮合的订单
    while unmatched_orders:
        unmatched = unmatched_orders.pop()
        curr_price = mkt_finder.find_mkt(curr_day, unmatched.ts_code, 'open')
        if curr_price is None:
            continue
        
        if unmatched.belong_trade is None:
            # 新开仓
            unclosed_trade = UnclosedTrade(open_order=unmatched.fill(curr_price, fee=fee))
            unclosed_trades.append(unclosed_trade)
            cash -= unclosed_trade.open_order.buy_amount
        else:
            # 平仓
            if unmatched.belong_trade not in unclosed_trades:
                # 已经被其他机制平掉了
                continue
            order = unmatched.fill(curr_price, fee=fee)
            trade = unmatched.belong_trade.close(order)
            finished_trades.append(trade)
            unclosed_trades.remove(unmatched.belong_trade)
            cash += trade.close_order.sell_amount

    # 记录资金曲线
    cash_curve.append({'date': curr_day, 'equity': cash + sum(ut.open_order.sell_amount for ut in unclosed_trades)})

    # <策略部分>
    # 开仓逻辑

    if trade_idx < len(trade_list_df):
        curr_trade_list = TradeList(**trade_list_df[trade_idx].to_dicts()[0])
        # print(curr_trade_list)
        if curr_trade_list['EntryTime'] != curr_day + timedelta(days=1):
            stock_len = 0
        else:
            stock_len = len(curr_trade_list['ts_code'])
            trade_idx += 1
    else:
        curr_trade_list = TradeList(
            EntryTime=curr_day + timedelta(days=1),
            name=[],
            ts_code=[],
            ps=[],
            ReturnPctList=[],
            num_trades=0,
            ret_5=[],
            total_mv=[],
            ReturnPct=0.0,
        )
        stock_len = 0

    for i in range(stock_len):
        single_stock_available_cash = cash * single_day_open_size / stock_len
        ts_code = curr_trade_list['ts_code'][i]
        name = curr_trade_list['name'][i]
        today_close = unwrap(mkt_finder.find_mkt(curr_day, ts_code, 'close'))

        entry_size = int(single_stock_available_cash / today_close)

        if entry_size == 0:
            print(f'[{curr_day}] {ts_code} {name} 资金不足，无法开仓 {cash=}')
            continue

        unmatched_orders.append(UnmatchedOrder(
            ts_code=ts_code,
            name=name,
            entry_time=curr_trade_list['EntryTime'],
            entry_size=entry_size,
            belong_trade=None,
        ))
        # print(f'\topen {unmatched_orders[-1]}')

    # 止损逻辑
    for unclosed_trade in unclosed_trades:
        today_price = mkt_finder.find_mkt(curr_day, unclosed_trade.ts_code, 'close')
        if today_price is None:
            continue
        if unclosed_trade.calc_return_pct(
            close_price=today_price,
            fee=fee,
        ) < stop_loss_p:
            unmatched_orders.append(UnmatchedOrder(
                ts_code=unclosed_trade.ts_code,
                name=unclosed_trade.name,
                entry_time=curr_trade_list['EntryTime'],
                entry_size=unclosed_trade.open_order.entry_size,
                belong_trade=unclosed_trade,
            ))
            # print(f'\tstop loss {unmatched_orders[-1]}')

    # </策略部分>

    # 然后检查是否有需要平仓的
    for unclosed_trade in unclosed_trades:
        if unclosed_trade.name not in curr_trade_list['name'] and curr_day - unclosed_trade.EntryTime > timedelta(days=5):
            unmatched_orders.append(UnmatchedOrder(
                ts_code=unclosed_trade.ts_code,
                name=unclosed_trade.name,
                entry_time=curr_trade_list['EntryTime'],
                entry_size=unclosed_trade.open_order.entry_size,
                belong_trade=unclosed_trade,
            ))
            # print(f'\tclose {unmatched_orders[-1]}')


# 将未开的订单取消, 未平的仓撤回
curr_day = all_trade_days[-1]
while unmatched_orders:
    unmatched = unmatched_orders.pop()
    curr_price = mkt_finder.find_mkt(curr_day, unmatched.ts_code, 'open')
    if curr_price is None:
        continue
    
    if unmatched.belong_trade is None:
        # 新开仓单取消
        pass
    else:
        # 平仓单撤回, 退钱
        unclosed_trades.remove(unmatched.belong_trade)
        cash += unmatched.belong_trade.open_order.buy_amount

# 收集数据
finished_trades_df = pl.DataFrame([
    {
        'ts_code': t.ts_code,
        'name': t.name,
        'EntryTime': t.EntryTime,
        'ExitTime': t.ExitTime,
        'EntryPrice': t.EntryPrice,
        'ExitPrice': t.close_order.entry_price,
        'Size': t.open_order.entry_size,
        'PnL': t.pnl,
        'ReturnPct': t.ReturnPct,
    }
    for t in finished_trades
])
# finished_trades_df
with pl.Config(tbl_cols=100, tbl_rows=21, float_precision=4):
    print(stat_df(finished_trades_df))  

print(f'final return_pct: {(cash_curve[-1]["equity"] / cash_init - 1) * 100:1.2f}%')
cash_curve_df = pl.DataFrame(cash_curve)
plt.plot(cash_curve_df['date'], cash_curve_df['equity'] / cash_init)

# %%
unmatched_orders, unclosed_trades, len(finished_trades)

# %%
plt.hist(finished_trades_df['ReturnPct'] * 100, bins=100)
plt.axvline(0, color='red', linestyle='--', linewidth=1)
# %%
[i for i in finished_trades if i.ReturnPct > 0.8]
# %%
[i for i in finished_trades if i.ReturnPct < -0.1]