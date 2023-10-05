import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

from numpy.linalg import LinAlgError
from scipy.linalg import pinv2 as pseudo_inverse


class FastWLS:
    def __init__(self, endog, exog, weight=None, pre_clean=True, **kwargs):
        x = exog
        y = endog
        w = np.ones_like(y) if weight is None else weight

        self.nrow, self.ncol = exog.shape
        self.column_mask = self.row_mask = None
        self.mask_idx = None

        if pre_clean:
            column_mask_temp = ~((0 == x) | np.isnan(x)).all(axis=0)
            x_col_filtered = x[:, column_mask_temp]

            y_mask = np.isnan(y)
            x_mask = np.isnan(x_col_filtered).any(axis=1)
            w_mask = np.isnan(w)

            self.row_mask = ~(x_mask | y_mask | w_mask)

            self.y = y[self.row_mask]
            self.w = w[self.row_mask]

            x_row_filtered = x[self.row_mask, :]
            self.column_mask = column_mask_temp & ~((0 == x_row_filtered) | np.isnan(x_row_filtered)).all(axis=0)

            self.x = x[self.row_mask, :][:, self.column_mask]
            self.mask_idx = np.ix_(self.row_mask, self.column_mask)

        else:
            self.x = x
            self.y = y
            self.w = w

        sqrt_w = np.sqrt(self.w)
        self.wy = sqrt_w * self.y
        self.wx = sqrt_w[:, np.newaxis] * self.x

        self.xtx = None
        self.beta = None
        self.fitted = None
        self.res = None

        self.nobs = self.wx.shape[0]
        self.rank = self.wx.shape[1]
        self.df_model = self.rank
        self.df_resid = self.nobs - self.rank

    def fit(self, regularization_L2=None):
        xtx = self.wx.T.dot(self.wx)
        xty = self.wx.T.dot(self.wy)
        if regularization_L2 is not None:
            xtx += regularization_L2[self.column_mask, :][:, self.column_mask]
        self.xtx = xtx
        try:
            self.beta = np.linalg.solve(xtx, xty)
        except LinAlgError:
            pinv = pseudo_inverse(xtx)
            self.beta = pinv @ xty
        self.fitted = self.x.dot(self.beta)
        self.res = self.y - self.fitted

    def resize_column_vector(self, data, subset_rng=None):
        if self.row_mask is None:
            return data

        mask = self.row_mask if subset_rng is None else self.row_mask[subset_rng]
        resized = np.empty(len(mask))
        resized.fill(np.nan)
        resized[mask] = data
        return resized

    @property
    def residuals(self):
        return self.resize_column_vector(self.res)


def ICIR_cal(sig_df, ret_df):
    corr_ = sig_df.T.corrwith(ret_df.T)
    ic = corr_.mean()
    ir = ic / corr_.std()
    rank_ic = sig_df.rank(1).T.corrwith(ret_df.rank(1).T).mean()
    return ic, ir, rank_ic


def alignAwithB(A, B, fill_method=None, fill_limit=0):
    A_ = A.reindex(A.index.union(B.index)).sort_index()
    A_ = A_.reindex(A.columns.union(B.columns), axis=1)
    if fill_method is not None and fill_limit > 0:
        A_ = A_.fillna(method=fill_method, limit=fill_limit)
    A_ = A_.loc[B.index, B.columns]
    return A_


def portfolio_hold_period(sig, rets, window_size=252):
    # Assume Signal and returns are correctly shifted.
    m = sig.copy()
    r = alignAwithB(rets, sig)
    denom = (m.fillna(0) - m.fillna(0).shift(1) * (1 + r.shift(1))).abs().sum(1).rolling(window_size,
                                                                                         min_periods=window_size // 2).mean()
    numer = m.abs().sum(1).rolling(window_size, min_periods=window_size // 2).mean()
    return 2 / denom.replace(0.0, np.nan) * numer


def PNL_analysis(sigs, ret_df, cutoff_dt: str = None,
                 nDaysYear=252, cost=0, nlargest_DD=3, detail_analysis=False):
    '''
    Perform PNL analysis on signal or portfolio
    :param sigs: in this case, should be shifted by 2 and also liquidated using the trading universe.
    :param ret_df: ret1d provided at date T.
    :param cutoff_dt: in-sample / out-sample cut, yyyy-mm-dd
    :param nDaysYear: how many trading dates a year
    :param cost: trading cost
    :param nlargest_DD: how many drawdowns want to be included.
    :param detail_analysis: True for portfolio analysis, False for signal selections
    :return:
    '''

    def _cal_pnl(pos, ret, cost=0, fill_limit=0):
        pos_reindexed = alignAwithB(pos, ret, fill_method='ffill',
                                    fill_limit=fill_limit)
        cost = pd.Series(cost, index=pos_reindexed.columns)
        pnls = (pos_reindexed * ret).fillna(0.0)  # According to instruction, no need to shift 1
        traded = pos_reindexed.fillna(0) - pos_reindexed.fillna(0).shift(1) * (1 + ret.shift(1))
        costs = traded.abs().mul(cost, 1)
        pnls = pnls.sub(costs, fill_value=0)
        pnls = pnls[pos.index[0]:pos.index[-1]]
        return pnls.sum(1)

    def _sharp_ratio(excess_ret, n):
        return excess_ret.mean() / excess_ret.std() * np.sqrt(n)

    def _cal_peaktothrough_drawdown(strategy_ret, n):
        val1 = strategy_ret.nsmallest(n, 'DrawDown')
        val1['DD_pct(%)'] = np.round(val1['DrawDown'] / val1['cummax'] * 100, 2)
        res = val1.reset_index().rename(columns={
            'date': 'DD_from_dt',
            'DATE': 'DD_to_dt'
        })
        return res

    def _cal_annual_ret(strategy_ret, col):
        annualized_return = strategy_ret.mean() * nDaysYear * 100
        volatility = strategy_ret.std() * np.sqrt(nDaysYear) * 100
        return annualized_return.values[0], volatility.values[0]

    pl_analysis = pd.concat([_cal_pnl(sigs[x], ret_df, cost=cost) for x in sigs.keys()], axis=1, keys=sigs.keys())

    hps = pd.concat([portfolio_hold_period(sigs[x], ret_df, window_size=nDaysYear) for x in sigs.keys()],
                    axis=1, keys=sigs.keys())

    sharp = _sharp_ratio(pl_analysis, n=nDaysYear)
    stats = pd.DataFrame({'Sharpe': sharp}).loc[list(sigs.keys())]
    start_dt = pl_analysis.index[0]
    end_dt = pl_analysis.index[-1]
    if cutoff_dt is not None:
        assert pd.to_datetime(cutoff_dt) > start_dt
        assert pd.to_datetime(cutoff_dt) < end_dt
        stats['IS Sharpe'] = _sharp_ratio(pl_analysis[:cutoff_dt], n=nDaysYear)
        stats['OOS Sharpe'] = _sharp_ratio(pl_analysis[cutoff_dt:], n=nDaysYear)
    stats['Daily TO(%)'] = 100 / hps.mean()

    cum_ret = pl_analysis.cumsum()

    if detail_analysis:
        # Calculate top3 drawdowns
        DrawDown_res = {}
        for col in cum_ret.columns:
            sret = cum_ret[[col]].copy()
            sret['cummax'] = sret[col].cummax()
            sret['date'] = sret.index
            sret = sret[[col, 'cummax']].merge(sret.groupby(['cummax'])[['date']].first().reset_index(),
                                               on='cummax', how='left')
            sret.index = cum_ret.index
            sret['DrawDown'] = sret[col] - sret['cummax']
            DrawDown_res[col] = _cal_peaktothrough_drawdown(sret, nlargest_DD)

            # Compute Cumulative Returns
            strategy_ret = pl_analysis[[col]].copy()
            start_pos = 0
            while strategy_ret.iloc[start_pos, 0] == 0.0:
                start_pos += 1
            strategy_ret = strategy_ret.iloc[start_pos:, :]
            ann_ret, ann_vol = _cal_annual_ret(strategy_ret, col)
            stats.loc[col, 'annualized_returns(%)'] = ann_ret
            stats.loc[col, 'annualized_volatility(%)'] = ann_vol

            # percentage of dates
            daily_pnl = pl_analysis[col]
            pos_dates = daily_pnl[daily_pnl > 0].shape[0]
            total_dates = daily_pnl.shape[0]
            stats.loc[col, 'positive_dates(%)'] = pos_dates * 100 / total_dates
        return stats, cum_ret, DrawDown_res
    return stats, cum_ret, pl_analysis


def show_drawdown(DrawDown_res):
    return pd.concat([v.rename(columns={k: 'cummulative_pnl'}, index={0: k}) for k, v in DrawDown_res.items()])


def const_gmv(sig):
    sig = sig.div(sig.abs().sum(1), axis=0)
    return sig


def xs_standardize(sig):
    sig_rank = sig.rank(1)
    sig_count = sig.count(1)
    sig_rank = sig_rank.div(sig_count + 1, 0)
    df = sig_rank.sub(sig_rank.mean(1), 0)
    df = df.div(df.std(1), 0)
    return df


def xs_normalize(sig):
    sig = sig.sub(sig.mean(1), 0)
    sig = sig.div(sig.std(1), 0)
    return sig


def display_stats(df):
    res = df.copy()
    for col in df.columns:
        res[col] = res[col].apply(lambda x: round(x, 2))
    return res


def factor_residualize(input_df,
                       factor_df,
                       add_constant=True):
    universe_df = input_df * 0 + 1
    factor_df = [alignAwithB(df, universe_df) * universe_df for df in factor_df]
    input_df = alignAwithB(input_df, universe_df) * universe_df

    if add_constant is True:
        factor_df.append(input_df * 0 + 1)

    def wls(endog, exog, weight=None):
        regr_model = FastWLS(endog=endog, exog=exog, weight=weight)
        regr_model.fit(regularization_L2=None)
        return getattr(regr_model, 'residuals')

    res = []
    for date in tqdm(universe_df.index, disable=False):
        y = input_df.loc[date]
        x = np.array([df.loc[date].values for df in factor_df]).T
        res.append([date] + list(wls(y.values, x, weight=None)))

    if len(res) == 0:
        return pd.DataFrame()
    indices = ['THIS_IS_DATE_COLUMN']
    indices += list(input_df.columns)
    df = pd.DataFrame(res, columns=indices).set_index('THIS_IS_DATE_COLUMN')
    df.index.name = None
    return df

