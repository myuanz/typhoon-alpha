# Typhoon Alpha

基于台风相关的百度搜索指数生成 Alpha, 之后基于横截面选股交易, 目前回测结果显示, 平均每个台风日可以获得 0.2% 的收益. 

![alt text](static/equity_curve.png)

## 复现

安装 [uv](https://github.com/astral-sh/uv), 然后

```bash
git clone https://github.com/myuanz/typhoon-alpha.git
cd typhoon-alpha
uv sync
```

仓库数据获取使用 tushare, 你也需要准备一个 tushare token.
