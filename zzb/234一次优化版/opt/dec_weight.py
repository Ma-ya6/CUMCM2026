# -*- coding: utf-8 -*-
"""五(a) 决策加权损失 与 五(b) 电价–净负荷联合预测。

## 为什么要换损失函数

上游预测层用**普通 MAE** 训练基模型、也用普通 MAE 决定三路集成的权重。但本题
的预测误差不是等价的：一个落在午间高电价时段的净负荷低估，会走到 **5 倍电价**
的紧急购电；同样的千瓦数落在夜间低谷，代价只有 $c_t$ 甚至 0。用 MAE 训练等于
**默认所有时段的误差一样贵**，模型会把容量花在"把夜间拟合得更好"上。

同理，集成权重用 MAE 选模型，等于让"夜间拟合得好的候选"与"正午拟合得好的
候选"平起平坐——而后者才是决定费用的那个。

## 决策权重怎么取

一阶近似：单位净负荷误差在时段 $t$ 的边际后果 $\\approx c_t$（多买/少买都按
电价计），而一旦触发紧急购电就放大到 $5c_t$。因此取**价格权重**

$$ w_{d,t} = \\frac{c_{d,t}}{\\bar c_d}, \\qquad \\bar c_d=\\tfrac1{144}\\sum_t c_{d,t}, $$

均值归一到 1，量纲与普通 MAE 一致，可直接对比。

在此之上加一个**方向惩罚**：净负荷**低估**与高估的后果并不对称——低估走到
5c 的紧急购电，高估只是多买（1c 或 1.5c）。故对"朝危险方向"的误差再乘
$1+\\beta$：

- 负载通道（``sign=+1``）：危险方向是**低估**（预测 < 实际，净负荷被低估）；
- 光伏通道（``sign=-1``）：危险方向是**高估**（预测 > 实际，净负荷同样被低估）。

## 因果性

$c_{d,t}$ 必须是第 $d$ 天 0:00 可知的：

- 问题3 的电价是附件1 的**固定曲线**，全程已知，$c_{d,t}$ 取真值；
- 问题4 的电价是未知量，取**前一日已实现曲线** $c_{d-1,t}$（第 0 天用附件1 的
  冷启动曲线）。

两者都只用 $j<d$ 的信息，故加权后的预测层仍然是因果的；``check_prefix.py``
的截断不变性检验对该预测层同样适用。

## 五(b) 联合建模

``extra_feature`` 钩子把**同一天的净负荷预测**作为额外一列喂进电价/负载的
回归，使模型能直接学到"高负荷 + 低光伏 + 高电价"同时出现的尾部，而不是
分别预测后在决策层才发现三者叠加。该列取**自有预测**（第 $d$ 天 0:00 已生成，
因果），不引入任何新信息。

## 与上游的等价性

``use_price_weight=False, beta=0.0, extra_feature=None`` 时本模块与上游
``seasonal.generate_causal_forecasts`` 走同一条算法路径，数值应**逐位一致**。
``run_dw.py`` 用这条等价性做自检：若不一致，说明差异来自实现而非加权本身，
A/B 结论不成立。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

N_SLOT = 144


def _stack():
    """取当前 sys.path 上的上游模块（多问题共用进程时必须延迟导入）。"""
    import deterministic_baseline as base
    from seasonal.online_correction import build_pipelines
    from seasonal.season_state import SeasonSensor
    return base, build_pipelines, SeasonSensor


def price_weight(price_day: np.ndarray) -> np.ndarray:
    """价格权重 $w_t=c_t/\\bar c$，均值归一到 1。"""
    c = np.asarray(price_day, dtype=float)
    mean = float(np.mean(c))
    if not np.isfinite(mean) or mean <= 0:
        return np.ones(N_SLOT)
    w = c / mean
    return np.where(np.isfinite(w), w, 1.0)


def price_by_day(data, *, use_forecast_price: bool) -> np.ndarray:
    """逐日"决策时可知"的电价曲线，形状 ``(n_days, 144)``。

    固定电价（问题2/3）直接取真值；逐日电价（问题4）取前一日已实现曲线，
    第 0 天退化为附件1 的冷启动曲线。
    """
    n = len(data.dates)
    if np.asarray(data.price).ndim == 1:
        return np.tile(np.asarray(data.price, dtype=float), (n, 1))
    price = np.asarray(data.price, dtype=float)
    out = np.empty((n, N_SLOT), dtype=float)
    initial = np.asarray(data.initial_price, dtype=float)
    out[0] = initial if np.asarray(initial).size == N_SLOT else price[0]
    out[1:] = price[:-1]
    return out


class CausalEnsembleDW:
    """上游 ``CausalEnsemble`` 的决策加权版本。

    与上游的三点差别，全部通过开关控制，关闭时逐位复刻上游：

    1. ``use_price_weight`` —— 误差度量由 $|\\hat y-y|$ 改为 $w_t|\\hat y-y|$；
    2. ``beta`` —— 对"朝危险方向"的误差再乘 $1+\\beta$；
    3. ``extra_feature`` —— 在特征矩阵末尾追加一列（如净负荷预测）。

    三者都会同时作用于**基模型拟合**（样本权重）与**集成加权**（误差历史），
    因为两处用的是同一个损失口径。
    """

    def __init__(
        self,
        *,
        values: np.ndarray,
        dates,
        initial: np.ndarray,
        baseline_fn,
        config,
        price_by_day: np.ndarray | None = None,
        use_price_weight: bool = True,
        beta: float = 0.0,
        sign: float = 1.0,
        extra_feature=None,
    ) -> None:
        self._base, _, self._SeasonSensor = _stack()
        self.values = values
        self.dates = dates
        self.initial = initial
        self.baseline_fn = baseline_fn
        self.config = config
        self.price_by_day = price_by_day
        self.use_price_weight = bool(use_price_weight) and price_by_day is not None
        self.beta = float(beta)
        self.sign = float(sign)
        self.extra_feature = extra_feature
        self.weights_by_day: list[dict] = []
        self._error_history: dict[str, list[float]] = {
            "baseline": [], "ridge": [], "lightgbm": []}
        self._ridge = None
        self._lgb = None
        self._last_fit_day: dict[str, int] = {"ridge": -10**9, "lightgbm": -10**9}
        self._stress_at_fit: dict[str, float] = {"ridge": 0.0, "lightgbm": 0.0}
        self._pending: dict[str, np.ndarray] = {}

    # ---- 权重与特征 -------------------------------------------------------
    def slot_weight(self, day: int) -> np.ndarray:
        """第 ``day`` 天的逐时段决策权重（均值归一到 1）。"""
        if not self.use_price_weight:
            return np.ones(N_SLOT)
        return price_weight(self.price_by_day[day])

    def _features(self, day: int) -> np.ndarray:
        f = self._base.day_features(self.values, self.dates, day)
        if self.extra_feature is not None:
            extra = np.asarray(self.extra_feature(day), dtype=float).reshape(-1, 1)
            f = np.column_stack([f, extra])
        return f

    def _training(self, start_day: int, end_day: int):
        days = range(max(14, start_day), end_day)
        x = np.vstack([self._features(day) for day in days])
        y = np.concatenate([self.values[day] for day in days])
        return x, y

    def _sample_weight(self, start_day: int, end_day: int, day: int):
        """训练样本权重：按目标时段的决策权重铺开（第 ``day`` 天的曲线）。

        未启用价格权重时返回 ``None``，使拟合与上游逐位一致（传全 1 权重会让
        LightGBM 走加权直方图分支，可能引入与"不加权"不同的浮点路径）。
        """
        if not self.use_price_weight:
            return None
        n_days = max(0, end_day - max(14, start_day))
        return np.tile(self.slot_weight(day), n_days)

    # ---- 与上游同构的滚动拟合 ---------------------------------------------
    def _compute_window(self, day: int, current_stress: float) -> int:
        cfg = self.config
        if day < cfg.ridge_min_day or not cfg.use_adaptive_window:
            return cfg.base_window_days
        window = int(np.clip(
            round(cfg.base_window_days / (1.0 + 2.5 * current_stress)),
            cfg.window_min_days, cfg.base_window_days))
        for label, period in (("ridge", cfg.ridge_refit_days),
                              ("lightgbm", cfg.lgb_refit_days)):
            if current_stress - self._stress_at_fit[label] >= cfg.stress_refit_jump:
                self._last_fit_day[label] = -10**9
        return window

    def _fit_lightgbm(self, day: int, window: int):
        base = self._base
        cfg = self.config
        start = max(14, day - window)
        split = max(start + 21, day - 21)
        x_tr, y_tr = self._training(start, split)
        x_va, y_va = self._training(split, day)
        sw_tr = self._sample_weight(start, split, day)
        sw_va = self._sample_weight(split, day, day)
        import lightgbm as lgb
        probe = base.make_lightgbm(600)
        probe.fit(x_tr, y_tr, sample_weight=sw_tr,
                  eval_set=[(x_va, y_va)], eval_sample_weight=[sw_va],
                  eval_metric="l1",
                  callbacks=[lgb.early_stopping(40, verbose=False),
                             lgb.log_evaluation(0)])
        best = int(np.clip(int(probe.best_iteration_ or 200), 30, 400))
        x_full, y_full = self._training(start, day)
        model = base.make_lightgbm(best)
        model.fit(x_full, y_full,
                  sample_weight=self._sample_weight(start, day, day))
        return model

    def predict(self, day: int, current_stress: float) -> np.ndarray:
        """只用 day 以前的数据生成第 day 天的集成预测（含决策加权）。"""
        base = self._base
        cfg = self.config
        window = self._compute_window(day, current_stress)
        candidates = {"baseline": self.baseline_fn(self.values, day, self.initial)}

        if day >= cfg.ridge_min_day:
            if self._ridge is None or day - self._last_fit_day["ridge"] >= cfg.ridge_refit_days:
                x_tr, y_tr = self._training(day - window, day)
                from sklearn.linear_model import Ridge
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                self._ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                self._ridge.fit(x_tr, y_tr,
                                ridge__sample_weight=self._sample_weight(day - window, day, day))
                self._last_fit_day["ridge"] = day
                self._stress_at_fit["ridge"] = current_stress
            candidates["ridge"] = self._ridge.predict(self._features(day))

        if day >= cfg.lgb_min_day:
            if self._lgb is None or day - self._last_fit_day["lightgbm"] >= cfg.lgb_refit_days:
                self._lgb = self._fit_lightgbm(day, window)
                self._last_fit_day["lightgbm"] = day
                self._stress_at_fit["lightgbm"] = current_stress
            candidates["lightgbm"] = self._lgb.predict(self._features(day))

        for name in candidates:
            candidates[name] = np.maximum(np.asarray(candidates[name], float), 0.0)

        weights = base.adaptive_error_weights(candidates.keys(), self._error_history)
        prediction = np.zeros(N_SLOT, float)
        for name, candidate in candidates.items():
            prediction = prediction + weights[name] * candidate
        self.weights_by_day.append(dict(weights))
        self._pending = candidates
        return np.maximum(prediction, 0.0)

    def observe(self, day: int) -> None:
        """登记各候选的**决策加权**当日误差，供次日加权。"""
        w = self.slot_weight(day)
        actual = self.values[day]
        for name, candidate in self._pending.items():
            err = np.abs(candidate - actual) * w
            if self.beta:
                wrong_way = (self.sign * (candidate - actual)) < 0.0
                err = err * (1.0 + self.beta * wrong_way)
            self._error_history[name].append(float(np.mean(err)))
        self._pending = {}


def generate_forecasts(
    data,
    config=None,
    *,
    price_by_day: np.ndarray | None = None,
    use_price_weight: bool = True,
    beta: float = 0.0,
    extra_load=None,
    extra_pv=None,
    verbose: bool = False,
) -> dict:
    """与上游 ``seasonal.generate_causal_forecasts`` 同构的逐日因果预测。

    开关全关（``use_price_weight=False, beta=0, extra_*=None``）时与上游逐位一致。
    """
    base, build_pipelines, SeasonSensor = _stack()
    cfg = config or _default_config()
    data = replace(
        data,
        load_kw=np.array(data.load_kw, dtype=float, copy=True),
        pv_kw=np.array(data.pv_kw, dtype=float, copy=True),
        price=np.array(data.price, dtype=float, copy=True),
    )
    n_days = len(data.dates)
    sensor = SeasonSensor(base.N_SLOT, cfg.drift)
    pipelines = build_pipelines(
        n_slot=base.N_SLOT, dt_hours=base.DT, sensor=sensor, config=cfg.correction)

    common = dict(config=cfg, price_by_day=price_by_day, use_price_weight=use_price_weight)
    load_ensemble = CausalEnsembleDW(
        values=data.load_kw, dates=data.dates, initial=data.initial_load,
        baseline_fn=base.weekly_load_baseline, sign=+1.0, beta=beta,
        extra_feature=extra_load, **common)
    pv_ensemble = CausalEnsembleDW(
        values=data.pv_kw, dates=data.dates, initial=data.initial_pv,
        baseline_fn=base.recent_pv_baseline, sign=-1.0, beta=beta,
        extra_feature=extra_pv, **common)

    out = {k: np.zeros_like(data.load_kw)
           for k in ("load", "pv", "load_base", "pv_base")}
    rows: list[dict] = []
    for day in range(n_days):
        date = data.dates[day]
        stress = sensor.detector.stress()
        base_load = load_ensemble.predict(day, stress)
        base_pv = pv_ensemble.predict(day, stress)
        outcome_load = pipelines["load"].apply(base_load, date)
        outcome_pv = pipelines["pv"].apply(base_pv, date)
        out["load_base"][day], out["pv_base"][day] = base_load, base_pv
        out["load"][day] = outcome_load.prediction
        out["pv"][day] = outcome_pv.prediction
        rows.append({"date": date, "day_index": day,
                     "season_stress": float(stress)})

        load_ensemble.observe(day)
        pv_ensemble.observe(day)
        pipelines["load"].update(base_load, data.load_kw[day], date, day)
        pipelines["pv"].update(base_pv, data.pv_kw[day], date, day)
        sensor.observe(date=date,
                       load_residual=data.load_kw[day] - outcome_load.prediction,
                       pv_actual_kw=data.pv_kw[day])
        if verbose and (day + 1) % 60 == 0:
            print(f"  预测进度：{day + 1}/{n_days} 天", flush=True)

    out["net"] = out["load"] - out["pv"]
    out["weights"] = {"load": load_ensemble.weights_by_day,
                      "pv": pv_ensemble.weights_by_day}
    import pandas as pd
    out["diagnostics"] = pd.DataFrame(rows)
    return out


def generate_price_forecast(
    data,
    diagnostics,
    config=None,
    *,
    price_by_day: np.ndarray | None = None,
    use_price_weight: bool = True,
    beta: float = 0.0,
    extra_feature=None,
) -> dict:
    """问题4 的电价因果预测，与 ``price_forecast.generate_causal_price_forecast`` 同构。

    电价本身不接季节校正管线，只用基础集成；``extra_feature`` 用于五(b) 的
    联合建模（把当日净负荷预测作为额外特征）。
    """
    base, _, _ = _stack()
    cfg = config or _default_config()
    stress = diagnostics.sort_values("day_index")["season_stress"].to_numpy(float)
    n_days = len(data.dates)
    if len(stress) != n_days:
        raise ValueError("诊断表长度与日期数不一致")

    ensemble = CausalEnsembleDW(
        values=np.array(data.price, dtype=float, copy=True),
        dates=data.dates,
        initial=np.array(data.initial_price, dtype=float, copy=True),
        baseline_fn=base.weekly_load_baseline,
        config=cfg,
        price_by_day=price_by_day,
        use_price_weight=use_price_weight,
        beta=beta,
        sign=+1.0,
        extra_feature=extra_feature,
    )
    forecast = np.zeros_like(data.price)
    for day in range(n_days):
        forecast[day] = ensemble.predict(day, float(stress[day]))
        ensemble.observe(day)
    mask = np.asarray(data.dates >= base.FORMAL_START)
    return {"price": forecast, "weights": ensemble.weights_by_day}


def _default_config():
    from seasonal import ForecastConfig
    return ForecastConfig()


# --------------------------------------------------------------------------
# 指标：把"普通 nMAE"与"决策加权 nMAE"并列，看换损失到底换来了什么
# --------------------------------------------------------------------------
def accuracy(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray,
             weight: np.ndarray | None = None) -> dict:
    """正式期的误差指标。``weight`` 给出时额外给出决策加权 nMAE。

    ``weight`` 可以是与 ``actual`` 同形的逐时段权重，也可以是长度为
    ``N_SLOT`` 的日内权重（自动广播到每一天）。决策加权 nMAE 的分母同样
    加权，保证与普通 nMAE 同量纲、可比。
    """
    a = np.asarray(actual, float)
    y = a[mask]
    p = np.asarray(predicted, float)[mask]
    err = np.abs(p - y)
    out = {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "nmae": float(err.mean() / max(np.abs(y).mean(), 1e-9)),
        "bias": float((p - y).mean()),
    }
    if weight is not None:
        w = np.asarray(weight, float)
        if w.shape != a.shape:
            w = np.broadcast_to(w, a.shape)
        w = w[mask]
        num = float(np.sum(w * err) / np.sum(w))
        den = float(np.sum(w * np.abs(y)) / np.sum(w))
        out["dw_mae"] = num
        out["dw_nmae"] = num / max(den, 1e-9)
    return out
