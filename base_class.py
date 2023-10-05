import pandas as pd
import numpy as np
import os


class Universe:
    def __init__(self, start_dt: str = None, end_dt: str = None):
        # Process dataset and combine them together
        d1 = pd.read_csv('security_reference_data_w_ret1d_1.csv')
        d2 = pd.read_csv('security_reference_data_w_ret1d_2.csv')
        self.rawDat = pd.concat([d1, d2])
        self.rawDat.data_date = pd.to_datetime(self.rawDat.data_date, format='%Y%m%d')
        self.rawDat.index = self.rawDat.data_date

        self.start_dt = self.rawDat.index.min().strftime('%Y-%m-%d') if start_dt is None else start_dt
        self.end_dt = self.rawDat.index.max().strftime('%Y-%m-%d') if end_dt is None else end_dt

        self.pivot_dat = self.rawDat.loc[self.start_dt: self.end_dt].pivot(columns='security_id',
                                                                           index='data_date')

    @property
    def trade_univ(self):
        return self.pivot_dat['in_trading_universe']

    @property
    def volume(self):
        return self.pivot_dat['volume']

    @property
    def group_id(self):
        return self.pivot_dat['group_id']

    @property
    def close_price(self):
        return self.pivot_dat['close_price']

    @property
    def ret_1d(self):
        return self.pivot_dat['ret1d']

    def get_level_id(self, level=1):
        num = int(8 - level)
        return self.group_id // 10 ** num


class Features:
    def __init__(self, indexname='data_date', colname='security_id'):
        self.raw_dfs = {}
        for i in range(1, 12):
            fn = f'data_set_{i}.csv'
            tmp_df = pd.read_csv(fn)
            tmp_df[indexname] = pd.to_datetime(tmp_df[indexname], format='%Y%m%d')
            self.raw_dfs[f'ds{i}'] = tmp_df.copy()
        self.pivot_dat = {}
        self.index_name = indexname
        self.colname = colname

    def try_pivot(self):
        for k, v in self.raw_dfs.items():
            cur_cols = v.columns
            try:
                if v.shape != v.drop_duplicates().shape:
                    print(f' -- potential issue when processing {k} -- ')

                self.pivot_dat[k] = v.drop_duplicates().pivot(index=self.index_name,
                                                              columns=self.colname, values=cur_cols[-1])
            except Exception as e:
                print(f' -- issue when processing {k} -- ')
                self.pivot_dat[k] = str(e)
        return


class RiskFactors:
    def __init__(self, start_dt: str = None, end_dt: str = None):
        # Process dataset and combine them together
        d1 = pd.read_csv('risk_factors_1.csv')
        d2 = pd.read_csv('risk_factors_2.csv')
        self.dat = pd.concat([d1, d2])
        self.dat.data_date = pd.to_datetime(self.dat.data_date, format='%Y%m%d')
        self.dat.index = self.dat.data_date

        self.start_dt = self.dat.index.min().strftime('%Y-%m-%d') if start_dt is None else start_dt
        self.end_dt = self.dat.index.max().strftime('%Y-%m-%d') if end_dt is None else end_dt

        self.pivot_dat = self.dat.loc[self.start_dt: self.end_dt].pivot(columns='security_id',
                                                                        index='data_date')

    def get_factor(self, name):
        assert name in self.dat.columns, 'invalid name'
        return self.pivot_dat[name]
