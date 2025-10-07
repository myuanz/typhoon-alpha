# %%
__import__("os").chdir(__import__("os").path.dirname(__file__))

import concurrent.futures
import datetime
import json
import multiprocessing
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
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

import akshare as ak
import lightgbm as lgb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import polars as pl
import polars_talib as plta
import statsmodels.api as sm
import tushare as ts
import tyro
import zhplot
from backtesting import Backtest, Strategy
from joblib import Parallel, delayed
from polars._typing import PolarsDataType
from scipy import stats
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import RegressionResultsWrapper
from tqdm import tqdm
from tushare.pro.client import DataApi

from typhoon_strategy import TyphoonEventStrategy
from utils import (
    cache_df,
    fetch_all_stock_list,
    fetch_daily_bar,
    fetch_daily_basic,
    set_cache_dir,
    to_pl_wrapper,
    with_try_wrapper,
)
from utils_typhoon import flat_typhoon_df, load_baidu_idnex, load_typhoon_df


def unwrap[T](x: T|None) -> T:
    assert x is not None
    return x

pc = pl.col


@dataclass
class Args:
    data_dir: Path
    result_dir: Path
    start_date: str = '20170501'
    '''yyyyMMdd'''
    end_date: str = '20250926'
    '''yyyyMMdd'''

    stock_list: Literal['fgdm', 'all'] = 'fgdm'
    print_summary: bool = False
    draw_plots: bool = False
    write_plots: bool = False


    cash: float = 1_0000_0000.0
    trade_fee: float = 0.0006
    '''万6'''

    mkt_x_cols: Sequence[str] = (
        '上证',
        '深成',
        '创业',
    )

    resid_x_cols: Sequence[str] = (
        '上证',
        '深成',
        '创业',
        'speed',
        'china_score',
        'index-台风',
        'index-台风路径',
        'index-暴雨',
        'index-暴雨预警',
    )

    # <超参数区>
    train_years: int = 2
    '''使用过去几年的数据训练模型'''
    min_days_for_training: int = 10
    '''训练模型所需的最少样本天数'''

    min_resid_r2: float = 0.1
    '''残差模型的最低 R² 要求'''

    trade_min_pred: float = 0.5
    trade_open_size: float = 0.5
    trade_hold_days: int = 0
    # </超参数区>

    @property
    def start_date_dt(self) -> datetime.date:
        return datetime.datetime.strptime(self.start_date, '%Y%m%d').date()
    @property
    def end_date_dt(self) -> datetime.date:
        return datetime.datetime.strptime(self.end_date, '%Y%m%d').date()

# %%
class TdfgAnalyze:
    def __init__(self, args: Args):
        self.args = args


        for p in (self.args.data_dir, self.args.result_dir):
            p.mkdir(exist_ok=True, parents=True)

        self.data_dir = self.args.data_dir
        self.result_dir = self.args.result_dir

        set_cache_dir(self.data_dir)

    def load_stock_list_df(self):
        all_stock_df = fetch_all_stock_list()

        wind_stock_df = pl.read_excel(self.data_dir / 'wind_input_stocks.xlsx').with_columns(
            symbol=pc('symbol').str.replace(r'\D+', '')
        )
        raw_df_length = len(wind_stock_df)

        wind_stock_df = wind_stock_df.join(all_stock_df, on='symbol', how='left').filter(
            pl.col('ts_code').is_null().not_()
        )

        if len(wind_stock_df) != raw_df_length:
            print(f'{len(wind_stock_df)} != {raw_df_length}')
        if (wind_stock_df['name'] == wind_stock_df['name_right']).all():
            print(f'{wind_stock_df["name"]} == {wind_stock_df["name_right"]}')

        wind_stock_df.drop_in_place('name_right')
        wind_stock_df.write_excel(self.result_dir / '10.wind_stock_df.xlsx')
        return all_stock_df, wind_stock_df
    
    def load_tdfg(self):
        tdfg_df = load_typhoon_df(self.data_dir / '台风.xlsx')
        tdfg_flat_df = flat_typhoon_df(tdfg_df)

        tdfg_df.write_excel(self.result_dir / '20.台风-norm.xlsx')
        tdfg_flat_df.write_excel(self.result_dir / '20.台风-flat.xlsx')

        return tdfg_flat_df

    def prepare_global_data(self):
        all_stock_df, wind_stock_df = self.load_stock_list_df()

        match self.args.stock_list:
            case 'all':
                self.stock_df = all_stock_df
            case 'fgdm':
                self.stock_df = wind_stock_df
            case _: raise Exception(f'Unknown stock_list: {self.args.stock_list}')

        self.tdfg_flat_df = self.load_tdfg()
        tdfg_baidu_index_df = load_baidu_idnex(self.data_dir / 'baidu_index')
        self.tdfg_flat_df = tdfg_baidu_index_df.join(self.tdfg_flat_df, on='date', how='left').fill_null(0)

        self.深成指数_pct = fetch_daily_bar('399001.SZ', '20050101', ta.args.end_date, is_index=True).select(
            'date',
            深成='pct_chg',
            深成_rsi14=plta.rsi(pc('close'), timeperiod=14).fill_nan(0),
            深成_close=pc('close')
        )
        self.创业板指_pct = fetch_daily_bar('399006.SZ', '20050101', ta.args.end_date, is_index=True).select(
            'date',
            创业='pct_chg',
            创业_rsi14=plta.rsi(pc('close'), timeperiod=14).fill_nan(0),
            创业_close=pc('close')
        )
        self.上证指数_pct = fetch_daily_bar('000001.SH', '20050101', ta.args.end_date, is_index=True).select(
            'date',
            上证='pct_chg',
            上证_rsi14=plta.rsi(pc('close'), timeperiod=14).fill_nan(0),
            上证_close=pc('close')
        )

    def regression_market(self, df: pl.DataFrame, x_cols: Iterable[str], print_summary: bool|None=None, train_on_next_day: bool=True) -> RegressionResultsWrapper:
        df_pd = df.select(
            'fgdm',
            'fgdm_next_day',
            *x_cols,
        ).to_pandas()

        X = sm.add_constant(df_pd[list(x_cols)])
        assert isinstance(X, pd.DataFrame)
        y = df_pd['fgdm']
        if train_on_next_day:
            y = df_pd['fgdm_next_day']
        model = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')

        if print_summary is None:
            print_summary = self.args.print_summary
        if print_summary: 
            print('mkt_model', model.summary())
        return model

    def regression_resid(self, df: pl.DataFrame, mkt_model: RegressionResultsWrapper, x_cols: Iterable[str], print_summary: bool|None=None) -> RegressionResultsWrapper:
        df_pd = df.select(*x_cols).to_pandas()

        X = sm.add_constant(df_pd, has_constant='add')
        y = mkt_model.resid

        model = sm.OLS(y, X, missing='drop').fit()

        if print_summary is None:
            print_summary = self.args.print_summary
        if print_summary: 
            print('resid_model', model.summary())
        return model

    def iter_speed_threshold_for_regression(self, df: pl.DataFrame, speed_thresholds: np.ndarray, resid_x_cols: Iterable[str], min_sample_days: int = 10, output_dir: Path|None=None) -> pl.DataFrame | None:
        res_rows = []
        for speed_threshold in speed_thresholds:
            df_filt = df.filter(
                pc('speed') > speed_threshold,
            )
            if len(df_filt) <= min_sample_days:
                continue

            mkt_model = self.regression_market(df_filt, x_cols=self.args.mkt_x_cols, print_summary=False)
            resid_model = self.regression_resid(df_filt, mkt_model, resid_x_cols, print_summary=False)

            trade_days = df_filt.select(max=pc('date').max(), min=pc('date').min()).select(pc('max') - pc('min')).item()
            assert isinstance(trade_days, datetime.timedelta)

            res_rows.append({
                'speed_thr': speed_threshold,
                'mkt_r2': mkt_model.rsquared,
                'resid_r2': resid_model.rsquared,
                'sample_days': len(df_filt),
                'sample_days_per_year': len(df_filt) / np.ceil(trade_days.days / 365),
            })
            if self.args.print_summary:
                print(f'{speed_threshold:.3f}: {len(df_filt)}days, resid_r2:{resid_model.rsquared:.3f}, mkt_r2:{mkt_model.rsquared:.3f}')

        if not res_rows:
            return None
        res_df = pl.DataFrame(res_rows)
        if self.args.draw_plots:
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            axes[0].plot(res_df['speed_thr'], res_df['mkt_r2'], label='base')
            axes[0].plot(res_df['speed_thr'], res_df['resid_r2'], label='resid_with_tdfg')
            axes[0].axhline(0.1, color='green', linestyle='--')

            axes[1].plot(res_df['speed_thr'], res_df['sample_days'])
            axes[1].axhline(80, color='red', linestyle='--')
            axes[1].set_xlabel('speed_thr')
            axes[1].set_ylabel('sample_days')

            axes[0].legend()
            axes[1].legend()
            if self.args.write_plots and output_dir is not None:
                plt.savefig(output_dir / f'30.回归-不同风速阈值对比.png')
            if self.args.draw_plots:
                plt.show()
            else:
                plt.close()
        return res_df

    def prepare_backtesting_data(
        self, 
        df: pl.DataFrame, 
        best_speed_thr: float, 
        resid_x_cols: Iterable[str]
    ):
        '''分年度计算市场和残差模型, 然后计算预测'''
        test_dfs = []
        fgdm_pred_totals = []
        metrics: dict[datetime.date, dict[str, float]] = {}
        df_filt = df.filter(pc('speed') >= best_speed_thr)

        curr_dt = datetime.date(2018, 1, 1)
        step_days = 7
        while curr_dt < self.args.end_date_dt:
            curr_dt += datetime.timedelta(days=step_days)

            train_df = df_filt.filter(
                (curr_dt - pc('date').dt.date()) > 0,
                (curr_dt - pc('date').dt.date()).dt.total_days() <= self.args.train_years * 365,
            ) # [curr_dt - train_years, curr_dt)
            if len(train_df) <= self.args.min_days_for_training:
                if self.args.print_summary:
                    print(f"{curr_dt}: train_df={len(train_df)}. data is too short")
                continue
            test_df = df_filt.filter(
                pc('date').dt.date() >= curr_dt,
                pc('date').dt.date() < (curr_dt + datetime.timedelta(days=step_days)),
            ) # [curr_dt, curr_dt + step_days)
            # print(f'[{curr_dt}] train_df: {train_df['date'].min()} - {train_df['date'].max()}, {len(train_df)} days; test_df: {test_df['date'].min()} - {test_df['date'].max()}, {len(test_df)} days')
            if len(test_df) == 0:
                if self.args.print_summary:
                    print(f"{curr_dt}: test_df={len(test_df)}. no data")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                mkt_model = self.regression_market(train_df, x_cols=self.args.mkt_x_cols, print_summary=False)
                resid_model = self.regression_resid(train_df, mkt_model, resid_x_cols, print_summary=False)
            # print(f'{year}: {mkt_model.params=}, {resid_model.params=}')
            try:
                # 数量太少的时候, sm.add_constant 不生效, 手工添加效果也不佳

                # 市场模型预测
                fgdm_X = sm.add_constant(test_df[self.args.mkt_x_cols].to_pandas(), has_constant='add')
                # print(year, fgdm_X.shape, fgdm_X.columns)
                fgdm_pred = mkt_model.predict(fgdm_X)
                # 残差模型预测
                resid_X = sm.add_constant(test_df[list(resid_x_cols)].to_pandas(), has_constant='add')
                # print(year, resid_X.shape, resid_X.columns, f'{resid_x_cols=}')
                resid_pred = resid_model.predict(resid_X)
            except Exception as e:
                if self.args.print_summary:
                    print(f"{curr_dt}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            assert isinstance(fgdm_pred, pd.Series) and isinstance(resid_pred, pd.Series)
            # fgdm_true = test_df['fgdm'].to_numpy()
            fgdm_pred = fgdm_pred.to_numpy()
            resid_pred = resid_pred.to_numpy()

            # 市场+台风残差
            fgdm_pred_total = fgdm_pred + resid_pred # type: ignore
            
            test_dfs.append(test_df)
            fgdm_pred_totals.append(fgdm_pred_total)

            # print(fgdm_true, fgdm_pred)
            metrics[curr_dt] = {
                'train_fgdm_r2': mkt_model.rsquared,
                'train_resid_r2': resid_model.rsquared,


                # 'test_fgdm_r2': r2_score(fgdm_true, fgdm_pred),
                # 'test_total_r2': r2_score(fgdm_true, fgdm_pred_total),
                # 'test_fgdm_mse': mean_squared_error(fgdm_true, fgdm_pred),
                # 'test_total_mse': mean_squared_error(fgdm_true, fgdm_pred_total),
                # 'test_sig_mse': mean_squared_error(np.sign(fgdm_true), np.sign(fgdm_pred_total)),
            }
        if not test_dfs:
            return
        test_df = pl.concat(test_dfs)
        fgdm_pred_total = np.concatenate(fgdm_pred_totals)
        fgdm_pred_total = pl.from_numpy(fgdm_pred_total, schema=['fgdm_pred_total']).with_columns(
            # fgdm_pred_total=pc('fgdm_pred_total').rolling_mean(5)
        )
        # print(df, df.columns)

        bt_df = pl.concat([
            test_df, 
            fgdm_pred_total,
        ], how='horizontal')
        bt_df = df.join(bt_df, on='date', how='left').fill_null(0)
        # print(f'{metrics=}')
        return bt_df, metrics

    def backtesting(self, bt_df: pl.DataFrame, output_dir: Path|None=None):
        if self.args.print_summary:
            print(f'{bt_df=}')
            print(f'{bt_df.columns=}')
        bt = Backtest(
            bt_df.to_pandas().set_index('date'), TyphoonEventStrategy, 
            cash=self.args.cash, 
            commission=self.args.trade_fee, 
            # exclusive_orders=True, # 开启后第二笔多单也会导致第一笔平仓
            # hedging=True,
            finalize_trades=True
        )
        stats = bt.run(
            min_p=self.args.trade_min_pred,
            open_size=self.args.trade_open_size,
            hold_days=self.args.trade_hold_days,
        )

        if self.args.print_summary:
            print(stats)
        if self.args.draw_plots:
            bt.plot(open_browser=False)
        if self.args.write_plots and output_dir is not None:
            bt.plot(filename=str(output_dir / f'50.回测-策略净值曲线.html'), open_browser=False)

        return bt, stats

    def collect_all_metrics(
        self, 
        stock_name: str, 
        ts_code: str,
        ohlcv_df: pl.DataFrame,
        bt_metrics: dict[datetime.date, dict[str, float]],
        thr_df: pl.DataFrame | None, 
        stats: pd.Series,
        output_dir: Path|None=None
    ):
        print(f'{ohlcv_df=}')
        # print(f'{ohlcv_df.columns=}')
        r2_mkt = self.regression_market(ohlcv_df, x_cols=self.args.mkt_x_cols).rsquared
        r2_mkt_with_tdfg = self.regression_market(ohlcv_df, x_cols=self.args.resid_x_cols).rsquared
        metrics = {
            'name': stock_name,
            'ts_code': ts_code,
            '上证-风电:r2': r2_mkt,
            '上证台风-风电:r2': r2_mkt_with_tdfg,
            'r2_diff': r2_mkt_with_tdfg - r2_mkt
        }
        if thr_df is not None:
            nice_records = thr_df.filter(pc('resid_r2') > self.args.min_resid_r2).sort('speed_thr')
            metrics['残差r2*样本天数'] = nice_records.select(
                s=pc('resid_r2') * pc('sample_days')
            ).sum().item()

        bt_metrics_df = pl.DataFrame([
            {'dt': dt.isoformat()} | m for dt, m in bt_metrics.items()
        ])
        if output_dir is not None:
            bt_metrics_df.write_excel(output_dir / f'60.回测-各年度指标.xlsx')

        metrics |= bt_metrics_df.mean().to_dicts()[0]

        stats_dict = dict(stats.items())
        for k in ('_strategy', '_trades', '_equity_curve'):
            stats_dict.pop(k)
        for k, v in stats_dict.items():
            if isinstance(v, pd.Timestamp):
                stats_dict[k] = v.to_pydatetime()
            if isinstance(v, pd.Timedelta):
                stats_dict[k] = f'{v.days}'

        stats._equity_curve.index.name = 'dt'
        year_return_df = pl.from_pandas(stats._equity_curve, include_index=True).sort('dt').group_by(pc('dt').dt.year()).agg(
            pc('Equity').first().alias('year_start_equity'),
            pc('Equity').last().alias('year_end_equity'),
        ).sort('dt').with_columns(
            year_equity_pct = (pc('year_end_equity') - pc('year_start_equity')) / pc('year_start_equity'),
        ).select(
            year='dt',
            year_return = 'year_equity_pct',
        )
        # print(year_return_df)
        year_return_dic = dict(zip(year_return_df['year'], year_return_df['year_return']))
        for year in range(2018, 2026):
            ret = year_return_dic.get(year, 0) * 100
            metrics[f'r_{year} %'] = ret
        # print(year_return_dic, metrics)

        metrics |= stats_dict
        if output_dir is not None:
            with open(output_dir / f'80-info.json', 'w') as f:
                f.write(orjson.dumps(metrics, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode('utf-8'))

        return metrics

    def fetch_ohlcv_with_typhoon(self, ts_code: str) -> pl.DataFrame | None:
        '''获取 OHLCV 数据, 包含台风、台风指数、上证指数等'''
        ohlcv_df = fetch_daily_bar(
            ts_code, '20050101', self.args.end_date
        )
        if len(ohlcv_df) == 0:
            return
        ohlcv_df = ohlcv_df.with_columns(
            fgdm_rsi14=plta.rsi(pc('close'), timeperiod=14).fill_nan(0),
            fgdm_next_day=pc('pct_chg').shift(-1),
        ).filter(
            pc('date') >= self.args.start_date_dt,
            pc('pct_chg').is_null().not_(),
            pc('fgdm_next_day').is_null().not_()
        ).select(
            'ts_code',
            'date',
            pl.col('open').alias('Open'),
            pl.col('high').alias('High'),
            pl.col('low').alias('Low'),
            pl.col('close').alias('Close'),
            fgdm='pct_chg',
            fgdm_next_day='fgdm_next_day',
            fgdm_rsi14='fgdm_rsi14',
        ).join(
            self.tdfg_flat_df, on='date', how='left'
        ).select(
            pl.all().exclude('ts_code_right')
        ).join(self.深成指数_pct, on='date').join(self.创业板指_pct, on='date').join(self.上证指数_pct, on='date')
        return ohlcv_df

    def process_stock(self, ts_code: str, output_dir: Path):
        if output_dir.joinpath('20-回归.txt').exists():
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        set_cache_dir(self.data_dir)

        ohlcv_df = self.fetch_ohlcv_with_typhoon(ts_code)
        if ohlcv_df is None:
            return

        thr_range = np.arange(0.5, 2.0, 0.01)
        thr_df = self.iter_speed_threshold_for_regression(ohlcv_df, thr_range, resid_x_cols=self.args.resid_x_cols, min_sample_days=self.args.min_days_for_training, output_dir=output_dir)
        # thr_df = None
        # if thr_df is None:
        #     raise Exception('training data is too short')

        # best_speed_thrs = thr_df.filter(pc('resid_r2') > self.args.min_resid_r2).sort('speed_thr')
        # if len(best_speed_thrs) == 0:
        #     raise Exception('no valid speed thresholds found')
        # best_speed_thr = float(best_speed_thrs[0, 'speed_thr'])

        best_speed_thr = 1
        if self.args.print_summary:
            print(f'best_speed_thr={best_speed_thr:.4f}')
        
        bt_data = self.prepare_backtesting_data(
            ohlcv_df,
            best_speed_thr,
            resid_x_cols=self.args.resid_x_cols,
        )
        if bt_data is None:
            raise Exception('no valid backtesting data')
        bt_df, metrics_by_period = bt_data
        # print(bt_df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bt, bt_stats = self.backtesting(bt_df, output_dir=output_dir)
        pl.from_pandas(bt_stats._trades).write_excel(output_dir / f'70.回测-交易明细.xlsx')

        all_metrics = self.collect_all_metrics(
            stock_name=output_dir.name,
            ts_code=ts_code,
            ohlcv_df=ohlcv_df.filter(pc('speed') >= best_speed_thr),
            bt_metrics=metrics_by_period,
            thr_df=thr_df,
            stats=bt_stats,
            output_dir=output_dir,
        )
        return all_metrics, bt_df, bt, bt_stats


if __name__ == '__main__':
    ta = TdfgAnalyze(Args(
        data_dir = Path('data/raw'),
        result_dir = Path(f'data/result-test-total-speed1'),
        # start_date = '20050101',
        # end_date = '20250926',
        # stock_list = 'fgdm',
        stock_list='all',
        print_summary=True,
        draw_plots=True,
        # write_plots=True,


        mkt_x_cols = (
            '上证', 
            '深成', 
            '创业',
            'fgdm_rsi14',

            # '上证_rsi14',
            # '深成_rsi14',
            # '创业_rsi14',

        ),

        resid_x_cols = (
            '上证',
            '深成',
            '创业',

            'fgdm_rsi14',
            # '上证_rsi14',
            # '深成_rsi14',
            # '创业_rsi14',

            'speed',
            'china_score',
            'index-台风',
            'index-台风路径',
            'index-暴雨',
            'index-暴雨预警',
        )
    ))
    ta.prepare_global_data()

    all_metrics, bt_df, bt, bt_stats = unwrap(ta.process_stock('836961.BJ', ta.args.result_dir / 'test'))
    print(bt_stats)
    print('sqn=', bt_stats['SQN'])
# %%
#     res = []
#     for row in tqdm(ta.stock_df.iter_rows(named=True)):

#         ohlcv_df = ta.fetch_ohlcv_with_typhoon(row['ts_code']).tail(-1).filter(
#             pc('date') != datetime.date(2025,4,7),
#             pc('speed') > 1
#         )
#         ohlcv_df
    
#         x_cols = [
#             '上证', 
#             '深成', 
#             '创业',
#             'fgdm_rsi14',

#             '上证_rsi14',
#             '深成_rsi14',
#             '创业_rsi14',
#         ]

#         X = sm.add_constant(ohlcv_df[x_cols].to_pandas())
#         y = ohlcv_df['fgdm_next_day'].to_pandas()
#         model = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')

#         # print('mkt_model', model.summary())

#         x_cols = [
#             '上证',
#             '深成',
#             '创业',

#             'fgdm_rsi14',
#             '上证_rsi14',
#             '深成_rsi14',
#             '创业_rsi14',

#             'speed',
#             'china_score',
#             'index-台风',
#             'index-台风路径',
#             'index-暴雨',
#             'index-暴雨预警',
#         ]

#         X = sm.add_constant(ohlcv_df[x_cols].to_pandas(), has_constant='add')
#         # y = ohlcv_df['fgdm'].shift(-1)[:-1].to_pandas()
#         y = model.resid
#         model2 = sm.OLS(y, X, missing='drop').fit()
#         # print('full_model', model2.summary())
#         res.append({
#             'name': row['name'],
#             'ts_code': row['ts_code'],
#             'r2_mkt': model.rsquared,
#             'r2_resid': model2.rsquared,
#         })
# # %%
#     r2_df = pl.DataFrame(res).sort('r2_resid', descending=True)
#     r2_df
# # %%
#     r2_df
# # %%
#     from seaborn import pairplot
#     pairplot(df_pd)
# %%
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     stats, heatmap = bt.optimize(
    #         min_p=np.arange(0, 3, 0.05).tolist(), 
    #         open_size=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
    #         hold_days=[0, 1, 2, 3, 5, 7],

    #         max_tries=2000,
    #         return_heatmap=True,
    #         # maximize='Equity Final [$]',
    #     )
    # print(stats)
    # print(f'{stats._strategy.hold_days=} {stats._strategy.min_p=} {stats._strategy.open_size=}')
    # bt.plot()

# # %%

#     hm = heatmap.groupby(['min_p', 'hold_days']).mean().unstack()
#     hm = hm[::-1]

#     fig, ax = plt.subplots(figsize=(4, 10))
#     im = ax.imshow(hm, cmap='viridis')
#     _ = (
#         ax.set_xticks(range(len(hm.columns)), labels=hm.columns),
#         ax.set_yticks(range(len(hm)), labels=hm.index),
#         ax.set_xlabel('open_size'),
#         ax.set_ylabel('min_p'),
#         ax.figure.colorbar(im, ax=ax),
#     )
# %%
    tasks = []

    for row in ta.stock_df.iter_rows(named=True):
        ts_code = row['ts_code']
        name = row['name']
        tasks.append(delayed(with_try_wrapper(ta.process_stock))(ts_code, ta.args.result_dir / name))

    results = Parallel(n_jobs=-1, return_as='generator')(tasks)

    for result in tqdm(results, total=len(tasks)):
        if result is None:
            continue
        all_metrics, bt_df, bt, bt_stats = unwrap(result)
        

    # %%

    all_infos = []
    for row in ta.stock_df.iter_rows(named=True):
        name = row['name']
        info_path = ta.args.result_dir / name / '80-info.json'
        if not info_path.exists():
            continue
        with open(info_path) as f:
            info = orjson.loads(f.read())
            all_infos.append(info)
    all_stock_df = ta.stock_df

    all_infos_df = ta.stock_df.join(
        pl.DataFrame(all_infos, infer_schema_length=100000).sort('Return (Ann.) [%]', descending=True).filter(
            pc('# Trades') > 5,
            pc('Duration').cast(pl.Int64) > 365,
        ),
        on='ts_code',
    ).select(
        pl.all().exclude('dt', 'ts_code_right', 'name_right', 'symbol', 'cnspell', 'market', 'act_name')
    )
    all_infos_df.write_excel(ta.args.result_dir / '90.所有股票指标汇总.xlsx')

    total_return = all_infos_df['Return [%]'].mean()
    buy_hold_return = all_infos_df['Buy & Hold Return [%]'].mean()
    avg_drawdown = all_infos_df['Avg. Drawdown Duration'].cast(pl.Int32).mean()
    avg_trade = all_infos_df['Avg. Trade [%]'].mean()

    t = f'总交易品种: {len(all_infos_df)} 总收益: {total_return:.2f}%, 买入持有收益: {buy_hold_return:.2f}%, 平均回撤天数: {avg_drawdown:.2f}, 每笔平均收益: {avg_trade:.2f}%'
    open(ta.args.result_dir / '90.所有股票指标汇总.txt', 'w').write(t)
    print(t)
    all_infos_df
# %%
    if 'get_ipython' not in globals():
        exit()

# %%
    from seaborn import pairplot
    cols = 'train_fgdm_r2	train_resid_r2\tCAGR [%]'.split('\t')

    draw_df = all_infos_df[cols].to_pandas()
    pairplot(draw_df)
# %%
    from sklearn.decomposition import PCA

    X = all_infos_df[cols]
    X = X.select(pl.all().exclude('CAGR [%]'))
    print(X)
    
    pca = PCA(n_components=2)
    pca.fit(X)
    print(sum(pca.explained_variance_ratio_))

    x_c2 = pca.transform(X)
    # x_c2 = X.to_numpy()
    c = draw_df['CAGR [%]']
    # c = np.clip(c * 10, 0, 200)
    plt.figure(figsize=(10, 10))
    plt.scatter(*x_c2.T, alpha=0.5, c=np.sign(c))
    plt.colorbar()
