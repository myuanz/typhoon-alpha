from functools import partial, wraps
import os
from pathlib import Path
from typing import (
    Callable,
    Literal,
    Mapping,
    ParamSpec,
    Sequence,
    TypeVar,
)
import pandas as pd
import polars as pl
import tushare as ts
from polars._typing import PolarsDataType


P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")  # 用于方法的 self/type 位置

def to_pl_wrapper(func: Callable[P, pd.DataFrame]) -> Callable[P, pl.DataFrame]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> pl.DataFrame:
        return pl.from_pandas(func(*args, **kwargs))
    return wrapper

def with_try_wrapper(func: Callable[P, R]) -> Callable[P, R | None]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f'Error in {func.__name__} with args={args}, kwargs={kwargs}: {e}')
            import traceback
            traceback.print_exc()
            return None
    return wrapper

def cache_df(name: str | Path | Callable[[tuple, dict], str], ext: Literal['csv', 'parquet'] = 'csv', force: bool = False, schema_overrides: Mapping[str, PolarsDataType] | Sequence[PolarsDataType] | None | None = None):
    def decorator(func: Callable[P, pl.DataFrame | pd.DataFrame]) -> Callable[P, pl.DataFrame]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> pl.DataFrame:
            if callable(name):
                name_str = name(args, kwargs)
            else:
                name_str = str(name)

            base_data_dir = Path(getattr(cache_df, '_base_data_dir', ''))
            p = base_data_dir / Path(f'{name_str}.{ext}')

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

def set_cache_dir(p: str|Path):
    setattr(cache_df, '_base_data_dir', p)

@cache_df(lambda args, kwargs: f'adj-hfq/daily_bar_{args[0]}_{args[1]}_{args[2]}', schema_overrides={
    'trade_date': pl.String,
    'date': pl.Date,
}, ext='parquet')
def fetch_daily_bar(ts_code: str, start_date: str, end_date: str, *, is_index: bool = False) -> pl.DataFrame:
    '''每日 bar, return columns: ts_code	trade_date	close	open	high	low	pre_close	change	pct_chg	vol	amount	date'''
    if is_index:
        func = ts.pro_api().index_daily
    else:
        func = partial(ts.pro_bar, adj='hfq', asset='E')
    try:
        return pl.from_pandas(func(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )).sort('trade_date').with_columns(
            date=pl.col('trade_date').str.to_date('%Y%m%d'),
        )
    except TypeError:
        return pl.DataFrame([])

@cache_df(lambda args, kwargs: f'basic/daily_data_{args[0]}_{args[1]}_{args[2]}', schema_overrides={
    'trade_date': pl.String,
    'date': pl.Date,
}, ext='parquet')
def fetch_daily_basic(
    ts_code: str,
    start_date: str,
    end_date: str,
    /,
) -> pl.DataFrame:
    """
    每日基本面数据
    Args:
        ts_code (str): TS股票代码 (Tushare stock code).
        start_date (str): 开始日期 (Start date) in 'YYYYMMDD' format.
        end_date (str): 结束日期 (End date) in 'YYYYMMDD' format.
        pro (DataApi): Tushare数据接口实例 (Tushare DataApi instance).
        is_index (bool, optional): 是否为指数 (Whether to fetch index data). Defaults to False.
    Returns:
        pl.DataFrame: A Polars DataFrame containing daily basic data with columns:
            - ts_code: 股票代码
            - trade_date: 交易日期
            - close: 当日收盘价
            - turnover_rate: 换手率（%）
            - turnover_rate_f: 换手率（自由流通股）
            - volume_ratio: 量比
            - pe: 市盈率（总市值/净利润，亏损的PE为空）
            - pe_ttm: 市盈率（TTM，亏损的PE为空）
            - pb: 市净率（总市值/净资产）
            - ps: 市销率
            - ps_ttm: 市销率（TTM）
            - dv_ratio: 股息率（%）
            - dv_ttm: 股息率（TTM）（%）
            - total_share: 总股本（万股）
            - float_share: 流通股本（万股）
            - free_share: 自由流通股本（万）
            - total_mv: 总市值（万元）
            - circ_mv: 流通市值（万元）
            - date: 日期 (datetime.date)
    """
    pro = ts.pro_api()
    return pl.from_pandas(pro.daily_basic(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
    )).sort('trade_date').with_columns(
        date=pl.col('trade_date').str.to_date('%Y%m%d'),
    )

@cache_df('all_stock_df', ext='csv', schema_overrides={
    'symbol': pl.String,
    'list_date': pl.String,
})
def fetch_all_stock_list() -> pl.DataFrame:
    '''获取所有股票列表, return columns: ts_code,symbol,name,area,industry,fullname,enname,market,exchange,curr_type,list_status,list_date,delist_date,is_hs'''
    pro = ts.pro_api()
    fields = 'ts_code,symbol,name,area,industry,fullname,enname,market,exchange,curr_type,list_status,list_date,delist_date,is_hs'
    上市df = pl.from_pandas(pro.stock_basic(exchange='', list_status='L', fields=fields))
    退市df = pl.from_pandas(pro.stock_basic(exchange='', list_status='D', fields=fields))
    return pl.concat([上市df, 退市df]).sort('ts_code')

if __name__ == '__main__':
    ts.set_token(os.getenv('TUSHARE_TOKEN', ''))
    set_cache_dir('data/raw')

    print(fetch_daily_bar('000001.SZ', '20050101', '20250926'))
    print(fetch_daily_bar('838670.BJ', '20050101', '20250926'))
    stock_list_df = fetch_all_stock_list()
    print(stock_list_df.group_by('list_status').agg(pl.count()).sort('list_status'))

    print(fetch_daily_basic('000001.SZ', '20050101', '20250926'))
