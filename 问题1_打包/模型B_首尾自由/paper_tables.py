# -*- coding: utf-8 -*-
"""问题一 · 生成论文表 1（指定时段购电量）与表 2（分时段充放电量）"""

import sys
import numpy as np
import pandas as pd

DT = 1.0 / 6.0


def main():
    csv_path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else ""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    seg = df[df["x_t (kW)"].notna()].reset_index(drop=True)
    E = df["E_t (kWh)"].values
    x = seg["x_t (kW)"].values * DT          # 每段购电量 kWh
    u = seg["u_t (kW)"].values * DT
    v = seg["v_t (kW)"].values * DT
    c = seg["c_t (元/kWh)"].values

    print("=" * 64)
    print("论文表 1 · %s" % name)
    print("=" * 64)
    print("  时间段            购电量/kWh")
    for m in [600, 720, 840, 960, 1080, 1200]:
        i = (m - 10) // 10
        print("  %02d:%02d-%02d:%02d     %10.4f"
              % (m // 60, m % 60, (m + 10) // 60, (m + 10) % 60, x[i]))
    Q = float(x.sum())
    C = float((c * x).sum())
    print("  全天购电量         %10.4f kWh" % Q)
    print("  全天购电费         %10.4f 元" % C)
    print("  平均购电价         %10.4f 元/kWh" % (C / Q))

    print()
    print("=" * 64)
    print("论文表 2 · %s" % name)
    print("=" * 64)
    print("  时间段         充电量/kWh      放电量/kWh")
    names = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
             "12:00-16:00", "16:00-20:00", "20:00-24:00"]
    for k, nm in enumerate(names):
        sl = slice(k * 24, (k + 1) * 24)
        print("  %-12s %12.4f %14.4f" % (nm, u[sl].sum(), v[sl].sum()))
    print("  0:00 储电量 %.4f kWh，24:00 储电量 %.4f kWh" % (E[0], E[-1]))

    print()
    print("  充电总量 %.4f kWh，放电总量 %.4f kWh，往返损耗 %.4f kWh"
          % (u.sum(), v.sum(), u.sum() - v.sum()))
    need = np.maximum(seg["净负荷 (kW)"].values * DT, 0.0)
    Cb = float((c * need).sum())
    print("  无储能基准购电量 %.4f kWh，购电费 %.4f 元" % (need.sum(), Cb))
    print("  储能节省 %.4f 元（%.4f%%）" % (Cb - C, (Cb - C) / Cb * 100))


if __name__ == "__main__":
    main()
