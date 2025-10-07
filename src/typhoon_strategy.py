import datetime

from backtesting import Strategy


class TyphoonEventStrategy(Strategy):
    min_p: float = 0.5
    open_size: float = 0.5
    hold_days: int = 0

    def init(self):
        self.last_open_dt = self.data.index[0]
        self.I(lambda : self.data.上证_close, name='上证', overlay=False)
        self.I(lambda : self.data.fgdm_pred_total, name='pred', overlay=False)
        for key in ['index-台风', 'index-台风路径', 'index-暴雨', 'index-暴雨预警', 'speed']:
            self.I(lambda : self.data[f'{key}'], name=key, overlay=False)

    def next(self):
        不应开仓 = datetime.datetime(2024, 9, 20) <= self.data.index[-1] <= datetime.datetime(2024, 9, 30)
        有钱开仓 = self._broker._cash > self.data.Close[-1] * self.open_size * 1.1

        pred = self.data.fgdm_pred_total[-1]
        date_diff = self.data.index[-1] - self.last_open_dt
        # print(f'[{self.data.index[-1]}] {pred=} {pred==0} {self.position=} {date_diff} {date_diff >= datetime.timedelta(days=2)=} {不应开仓=} {有钱开仓=} {self._broker._cash=} {pred == 0 and self.position=}')

        if pred > self.min_p:
            if not 不应开仓 and 有钱开仓:
                self.buy(size=self.open_size)
                self.last_open_dt = self.data.index[-1]
        elif pred < -self.min_p:
            self.position.close()

        if (pred == 0 and self.position and date_diff >= datetime.timedelta(days=float(self.hold_days))) or 不应开仓:
            # print('close')
            self.position.close()
