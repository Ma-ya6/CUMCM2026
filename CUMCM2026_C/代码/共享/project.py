"""项目级入口：全年计算、核验、出图数据。模型源码只在 代码/共享 与各问题 deps 中保留一份。"""
from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # 代码/共享
PROJECT = ROOT.parent.parent                    # 项目根
VERIFY = PROJECT / "验证" / "代码"
PROBLEMS = ('Q2', 'Q3', 'Q4-2', 'Q4-3')


def call(script, *args):
    subprocess.run([sys.executable, str(script), *args], cwd=script.parent, check=True)


def verify(deep=False):
    call(VERIFY / "verify_results.py")
    call(VERIFY / "test_time_rules.py")
    call(VERIFY / "audits" / "finalize_df.py")
    if deep:
        call(VERIFY / "audits" / "logic_audit.py")


def df_ready():
    """表D/表F 由 出图数据/run_experiments.py 按需生成（约 38 次全年仿真）。"""
    tables = PROJECT / "图表" / "数据表"
    return ((tables / "表F_全知下界阶梯_正式期334天.csv").exists()
            and len(list(tables.glob("*_表D_分位数扫描_正式期334天.csv"))) == 4)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run'); run.add_argument('--problem', choices=(*PROBLEMS, 'Q4'))
    smoke = sub.add_parser('smoke'); smoke.add_argument('--problem', choices=PROBLEMS, default='Q2')
    smoke.add_argument('--days', type=int, default=2)
    check = sub.add_parser('verify'); check.add_argument('--deep', action='store_true')
    sub.add_parser('export')
    args = ap.parse_args()
    if args.command == 'run':
        for p in (('Q4-2', 'Q4-3') if args.problem == 'Q4' else ((args.problem,) if args.problem else PROBLEMS)):
            call(ROOT / 'run_year.py', '--problem', p)
    elif args.command == 'smoke':
        if not 1 <= args.days <= 365:
            ap.error('--days must be between 1 and 365')
        import run_year as model
        # 避免 run_year --days 覆盖已交付的全年 CSV。
        original = model.OUT_OVERRIDE
        try:
            model.OUT_OVERRIDE = Path(tempfile.mkdtemp(prefix='smoke-'))
            model.simulate(model.prepare(), args.problem, args.days)
            print('Isolated smoke artifacts:', model.OUT_OVERRIDE)
        finally:
            model.OUT_OVERRIDE = original
    elif args.command == 'verify':
        verify(args.deep)
    elif args.command == 'export':
        call(ROOT / "出图数据" / "export_plot_data.py")
        if df_ready():
            call(VERIFY / "audits" / "finalize_df.py")
        else:
            print("缺 表D/表F：跳过 D/F 完成度核验；X-1—X-4 与 Y-1 也将跳过。")
            print("补算：python 代码/共享/出图数据/run_experiments.py all ，然后再跑 project.py export")
        call(ROOT / "出图数据" / "plot_result_figures.py")
        call(ROOT / "出图数据" / "综合图.py")


if __name__ == '__main__':
    main()
