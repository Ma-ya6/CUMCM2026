# -*- coding: utf-8 -*-
"""从冻结 CSV 生成论文3 指定日期表格片段（写入 _build/src/_tables/*.md）。"""
import os, pandas as pd, numpy as np

ROOT = r"D:\Files\Work\Mathematical_Modeling\GuoSai\2026\C题\CUMCM2026\CUMCM2026_C"
TB = os.path.join(ROOT, '图表', '数据表')
RES = os.path.join(ROOT, '结果')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', '_tables')
os.makedirs(OUT, exist_ok=True)

DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
LBL = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
SEGS = ['10:00-10:10', '12:00-12:10', '14:00-14:10', '16:00-16:10', '18:00-18:10', '20:00-20:10']
BLK = [('00:00','04:00'),('04:00','08:00'),('08:00','12:00'),('12:00','16:00'),('16:00','20:00'),('20:00','24:00')]

TAG = {
    'Q2': ('Q2_reserve-mpc', '问题2/Q2_physical_10min.csv', 'fixed'),
    'Q3': ('Q3_exact-greedy', '问题3/Q3_physical_10min.csv', 'fixed'),
    'Q4-2': ('Q4-2_reserve-mpc', '问题4/问题4-2/Q4-2_physical_10min.csv', 'vary'),
    'Q4-3': ('Q4-3_exact-greedy', '问题4/问题4-3/Q4-3_physical_10min.csv', 'vary'),
}


def f2(x):
    return '{:,.2f}'.format(x)


def data(prob):
    tag, phys, price_kind = TAG[prob]
    t1 = pd.read_csv(os.path.join(TB, '%s_指定日期_表1.csv' % tag)).set_index('date')
    t2 = pd.read_csv(os.path.join(TB, '%s_指定日期_表2.csv' % tag)).set_index('date')
    t3 = pd.read_csv(os.path.join(TB, '%s_指定日期_表3.csv' % tag))
    ph = pd.read_csv(os.path.join(RES, phys))
    ph = ph[ph.date.isin(DATES)]
    out = {}
    for d in DATES:
        r1, r2 = t1.loc[d], t2.loc[d]
        p = ph[ph.date == d].sort_values('physical_slot0')
        emer_kwh = float(p.emergency.sum())
        emer_cost = float((5.0 * p.price_actual * p.emergency).sum())
        daily = dict(
            seg=[float(r1['%s_purchase_kwh' % s]) for s in SEGS],
            plan_first=float(r1.planned_purchase_kwh),
            plan_final=float(r1.final_commitment_kwh),
            cost=float(r1.total_cost_yuan),
            emer_kwh=emer_kwh, emer_cost=emer_cost,
            soc0=float(r2.soc_00_kwh), soc24=float(r2.soc_24_kwh),
            chg=[float(p.charge.values[i*24:(i+1)*24].sum()) for i in range(6)],
            dis=[float(p.discharge.values[i*24:(i+1)*24].sum()) for i in range(6)],
            chg_tot=float(p.charge.sum()), dis_tot=float(p.discharge.sum()),
        )
        out[d] = daily
    return out


def emit(prob, fname, title, note, base_extra=None):
    D = data(prob)
    lines = []
    lines.append('@@TBL')
    lines.append('caption: %s' % title)
    lines.append('cols: 项目 | ' + ' | '.join(LBL))
    for i, s in enumerate(SEGS):
        lines.append('row: 时段 %s 购电量 / kWh | ' % s + ' | '.join(f2(D[d]['seg'][i]) for d in DATES))
    lines.append('row: 全天首次计划购电量 / kWh | ' + ' | '.join(f2(D[d]['plan_first']) for d in DATES))
    lines.append('row: 全天最终有效承诺购电量 / kWh | ' + ' | '.join(f2(D[d]['plan_final']) for d in DATES))
    if base_extra:
        for d in DATES:
            pass
    lines.append('row: 全天购电费 / 元 | ' + ' | '.join(f2(D[d]['cost']) for d in DATES))
    lines.append('row: 当天紧急购电量 / kWh | ' + ' | '.join(f2(D[d]['emer_kwh']) for d in DATES))
    lines.append('row: 当天紧急购电费 / 元 | ' + ' | '.join(f2(D[d]['emer_cost']) for d in DATES))
    lines.append('row: 0:00 储电量 / kWh | ' + ' | '.join(f2(D[d]['soc0']) for d in DATES))
    lines.append('row: 24:00 储电量 / kWh | ' + ' | '.join(f2(D[d]['soc24']) for d in DATES))
    for i, (a, b) in enumerate(BLK):
        lines.append('row: %s—%s 充电量 / kWh | ' % (a, b) + ' | '.join(f2(D[d]['chg'][i]) for d in DATES))
        lines.append('row: %s—%s 放电量 / kWh | ' % (a, b) + ' | '.join(f2(D[d]['dis'][i]) for d in DATES))
    lines.append('row: 全天充电量 / kWh | ' + ' | '.join(f2(D[d]['chg_tot']) for d in DATES))
    lines.append('row: 全天放电量 / kWh | ' + ' | '.join(f2(D[d]['dis_tot']) for d in DATES))
    lines.append('note: %s' % note)
    lines.append('@@ENDTBL')
    with open(os.path.join(OUT, fname), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', fname)


def emit_q4(fname):
    """问题四：4-2 与 4-3 并排。"""
    A, B = data('Q4-2'), data('Q4-3')
    lines = ['@@TBL']
    lines.append('caption: 表9 问题 4-2 与 4-3 在指定时间段的购电量、全天购电费及紧急购电情况（正式评价期）')
    lines.append('cols: 项目 | ' + ' | '.join(['4-2 ' + d[5:] for d in DATES] + ['4-3 ' + d[5:] for d in DATES]))
    for i, s in enumerate(SEGS):
        lines.append('row: 时段 %s 购电量 / kWh | ' % s +
                     ' | '.join(f2(A[d]['seg'][i]) for d in DATES) + ' | ' +
                     ' | '.join(f2(B[d]['seg'][i]) for d in DATES))
    lines.append('row: 全天首次计划购电量 / kWh | ' +
                 ' | '.join(f2(A[d]['plan_first']) for d in DATES) + ' | ' +
                 ' | '.join(f2(B[d]['plan_first']) for d in DATES))
    lines.append('row: 全天最终有效承诺购电量 / kWh | ' +
                 ' | '.join(f2(A[d]['plan_final']) for d in DATES) + ' | ' +
                 ' | '.join(f2(B[d]['plan_final']) for d in DATES))
    lines.append('row: 全天购电费 / 元 | ' +
                 ' | '.join(f2(A[d]['cost']) for d in DATES) + ' | ' +
                 ' | '.join(f2(B[d]['cost']) for d in DATES))
    lines.append('row: 当天紧急购电量 / kWh | ' +
                 ' | '.join(f2(A[d]['emer_kwh']) for d in DATES) + ' | ' +
                 ' | '.join(f2(B[d]['emer_kwh']) for d in DATES))
    lines.append('row: 当天紧急购电费 / 元 | ' +
                 ' | '.join(f2(A[d]['emer_cost']) for d in DATES) + ' | ' +
                 ' | '.join(f2(B[d]['emer_cost']) for d in DATES))
    lines.append('row: 0:00 储电量 / kWh | ' +
                 ' | '.join(f2(A[d]['soc0']) for d in DATES) + ' | ' +
                 ' | '.join(f2(B[d]['soc0']) for d in DATES))
    lines.append('row: 24:00 储电量 / kWh | ' +
                 ' | '.join(f2(A[d]['soc24']) for d in DATES) + ' | ' +
                 ' | '.join(f2(B[d]['soc24']) for d in DATES))
    for i, (a, b) in enumerate(BLK):
        lines.append('row: %s—%s 充电量 / kWh | ' % (a, b) +
                     ' | '.join(f2(A[d]['chg'][i]) for d in DATES) + ' | ' +
                     ' | '.join(f2(B[d]['chg'][i]) for d in DATES))
        lines.append('row: %s—%s 放电量 / kWh | ' % (a, b) +
                     ' | '.join(f2(A[d]['dis'][i]) for d in DATES) + ' | ' +
                     ' | '.join(f2(B[d]['dis'][i]) for d in DATES))
    lines.append('row: 全天充电量 / kWh | ' +
                 ' | '.join(f2(A[d]['chg_tot']) for d in DATES) + ' | ' +
                 ' | '.join(f2(B[d]['chg_tot']) for d in DATES))
    lines.append('row: 全天放电量 / kWh | ' +
                 ' | '.join(f2(A[d]['dis_tot']) for d in DATES) + ' | ' +
                 ' | '.join(f2(B[d]['dis_tot']) for d in DATES))
    lines.append('note: 注：购电量均为最终有效承诺量。全天购电费按题面结算口径给出，含基准费、'
                 '调整费与紧急购电费；4-2 当天仅一次计划，最终有效承诺量等于首次计划量。')
    lines.append('@@ENDTBL')
    with open(os.path.join(OUT, fname), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', fname)


emit('Q2', 't_q2.md',
     '表7 问题二在指定时间段的购电量、全天购电费及紧急购电情况（正式评价期）',
     '注：六个指定时段的购电量与全天购电量均为最终有效承诺量；问题二每天仅制定一次计划，'
     '最终有效承诺量等于 00:00 首次计划量。全天购电费含计划购电费与紧急购电费。')
emit('Q3', 't_q3.md',
     '表8 问题三在指定时间段的购电量、全天购电费及紧急购电情况（正式评价期）',
     '注：六个指定时段的购电量与「全天最终有效承诺购电量」为当日 18:00 修订后的最终承诺量；'
     '「全天首次计划购电量」为 00:00 首次计划量。全天购电费按题面结算口径给出，'
     '基准费以路径最小承诺结算基准量计价，另叠加下调费、上调费与紧急购电费，'
     '故不等于任一承诺量乘以电价。')
emit_q4('t_q4.md')

# ---- 表H 派生：发布时点价值
h = pd.read_csv(os.path.join(TB, '表H_发布时点消融_正式期334天.csv'))
rows = []
for p in ['Q3', 'Q4-3']:
    d = h[h.problem == p].set_index('variant')
    full = d.loc['full']
    rows.append((p, d, full))
with open(os.path.join(OUT, 't_h.md'), 'w', encoding='utf-8') as f:
    L = ['@@TBL',
         'caption: 表10 发布时点消融：正式评价期 334 天各时点启用组合与总费用',
         'cols: 启用时点 | 问题三费用 / 元 | 问题三相对完整配置 | 问题 4-3 费用 / 元 | 问题 4-3 相对完整配置']
    NAMES = [('base_0', '仅 0:00'), ('only_6', '0:00 + 6:00'), ('only_12', '0:00 + 12:00'),
             ('only_18', '0:00 + 18:00'), ('cum_0612', '0:00 + 6:00 + 12:00（关闭 18:00）'),
             ('drop_6', '0:00 + 12:00 + 18:00（关闭 6:00）'),
             ('drop_12', '0:00 + 6:00 + 18:00（关闭 12:00）'),
             ('full', '0:00 + 6:00 + 12:00 + 18:00（完整配置）')]
    for key, name in NAMES:
        a = rows[0][1].loc[key].formal_period_cost_yuan
        b = rows[1][1].loc[key].formal_period_cost_yuan
        fa = rows[0][2].formal_period_cost_yuan
        fb = rows[1][2].formal_period_cost_yuan
        da = a - fa
        db = b - fb
        sa = '—' if key == 'full' else '{:+,.0f} 元'.format(da)
        sb = '—' if key == 'full' else '{:+,.0f} 元'.format(db)
        L.append('row: %s | %s | %s | %s | %s' % (name, f2(a).replace('.00', ''), sa,
                                                  f2(b).replace('.00', ''), sb))
    L.append('note: 注：相对完整配置 = 该组合费用 − 完整配置费用，负值表示比完整配置更省。'
             '表中全部组合均由同一套代码在正式评价期 334 天上回放得到，未重新调参。')
    L.append('@@ENDTBL')
    f.write('\n'.join(L) + '\n')
print('wrote t_h.md')

# ---- 表D 分位数扫描
D = {}
for q, name in [('Q2', 'Q2_reserve-mpc'), ('Q3', 'Q3_exact-greedy'),
                ('Q4-2', 'Q4-2_reserve-mpc'), ('Q4-3', 'Q4-3_exact-greedy')]:
    D[q] = pd.read_csv(os.path.join(TB, '%s_表D_分位数扫描_正式期334天.csv' % name)).set_index('q')
with open(os.path.join(OUT, 't_d.md'), 'w', encoding='utf-8') as f:
    L = ['@@TBL',
         'caption: 表12 风险分位数灵敏度扫描（正式评价期 334 天，事后对照，不参与参数选择）',
         'cols: 分位数 q | 问题二费用 / 元 | 问题二应急电量 / kWh | 问题三费用 / 元 | 问题三应急电量 / kWh | 问题 4-2 费用 / 元 | 问题 4-2 应急电量 / kWh | 问题 4-3 费用 / 元 | 问题 4-3 应急电量 / kWh']
    for qv in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        cells = ['%.2f' % qv]
        for k in ['Q2', 'Q3', 'Q4-2', 'Q4-3']:
            r = D[k].loc[qv]
            cells.append(f2(r.total_cost_yuan).replace('.00', ''))
            cells.append(f2(r.emergency_kwh).replace('.00', ''))
        L.append('row: ' + ' | '.join(cells))
    L.append('note: 注：0.80 为事前由报童临界比确定的取值，本表仅作事后敏感性对照。'
             '扫描最低点只在本表所列档位内成立。')
    L.append('@@ENDTBL')
    f.write('\n'.join(L) + '\n')
print('wrote t_d.md')

# ---- 表G 最终档位指标
g = pd.read_csv(os.path.join(TB, '表G_最终档位指标汇总_正式期334天.csv')).set_index('问题')
with open(os.path.join(OUT, 't_g.md'), 'w', encoding='utf-8') as f:
    L = ['@@TBL',
         'caption: 表6 四个（子）问题正式评价期 334 天的全年运行结果',
         'cols: 项目 | 问题二 | 问题三 | 问题 4-2 | 问题 4-3']
    def r(k, col, fmt='{:,.0f}'):
        return ' | '.join(fmt.format(g.loc[k, col]) for k in ['Q2', 'Q3', 'Q4-2', 'Q4-3'])
    L.append('row: 总费用 / 元 | ' + r(None, 'total_cost_yuan', '{:,.2f}'))
    L.append('row: 紧急购电量 / kWh | ' + r(None, 'emergency_kwh', '{:,.0f}'))
    L.append('row: 紧急购电天数 / 天 | ' + r(None, 'days_with_emergency', '{:,.0f}'))
    L.append('row: 紧急购电量占负载比 | ' +
             ' | '.join('{:.3%}'.format(g.loc[k, 'emergency_ratio_of_load']) for k in ['Q2','Q3','Q4-2','Q4-3']))
    L.append('row: 未用计划电量 / kWh | ' + r(None, 'unused_plan_kwh', '{:,.0f}'))
    L.append('row: 弃光电量 / kWh | ' + r(None, 'curtailed_pv_kwh', '{:,.0f}'))
    L.append('row: 全年调整电量 / kWh | ' + r(None, 'adjusted_kwh', '{:,.0f}'))
    L.append('row: 日费用 CVaR95 / 元 | ' + r(None, 'daily_cost_cvar95_yuan', '{:,.0f}'))
    L.append('row: 安全储备年均值 / kWh | ' + r(None, 'reserve_mean_kwh', '{:,.2f}'))
    L.append('row: 年末储电量 / kWh | ' + r(None, 'final_soc_kwh', '{:,.1f}'))
    L.append('note: 注：正式评价期 2025-02-01 至 2025-12-31 共 334 天。总费用为最终结算总额；'
             '四个（子）问题均以年末储电量 6000 kWh 收口。')
    L.append('@@ENDTBL')
    f.write('\n'.join(L) + '\n')
print('wrote t_g.md')


# ---- 表F 全知对照阶梯
f = pd.read_csv(os.path.join(TB, '表F_全知下界阶梯_正式期334天.csv'), encoding='utf-8-sig').set_index('问题')
with open(os.path.join(OUT, 't_f.md'), 'w', encoding='utf-8') as fh:
    L = ['@@TBL',
         'caption: 表11 全知对照阶梯：正式评价期 334 天四个档位的总费用（万元）',
         'cols: 问题 | C0 物理松弛下界 | C1 负荷、光伏与电价均已知 | C2 仅电价已知 | C3 本文完整策略']
    NAME = {'Q2': '问题二', 'Q3': '问题三', 'Q4-2': '问题 4-2', 'Q4-3': '问题 4-3'}
    for k in ['Q2', 'Q3', 'Q4-2', 'Q4-3']:
        r = f.loc[k]
        L.append('row: %s | %.2f | %.2f | %.2f | %.2f' % (NAME[k], r.C0, r.C1, r.C2, r.C3))
    L.append('note: 注：C0 为把正式期初始储电量松弛到 10800 kWh（储能上界）后、'
             '负荷、光伏与电价均取真实值的物理可行松弛解，只用于给出量级参照，不对应任何可实施的策略。'
             'C1 的负荷、光伏与电价均为真实值；C2 只把电价换成真实值，负荷与光伏仍用预测值；'
             'C3 全部用历史信息预测。三档均保持本文采用的执行方式、储备设置与年末收口条件，'
             '是同一套代码在 334 天上回放的结果，并非各信息条件下的年度最优解。'
             '因此 C2−C1 度量负荷与光伏预测误差的代价、C3−C2 度量电价预测误差的代价。'
             '问题二、问题三的电价为固定值，C2 与 C3 相等。四个档位之间的差额是含符号的启发式差距，'
             '不能逐项认定为某一模块的因果贡献。')
    L.append('@@ENDTBL')
    fh.write('\n'.join(L) + '\n')
print('wrote t_f.md')
