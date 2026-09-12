"""Project-level run, verification and packaging; shared model source stays unique."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROBLEMS=('Q2','Q3','Q4-2','Q4-3')
GROUPS={'问题2最终模型':('Q2',),'问题3最终模型':('Q3',),'问题4最终模型':('Q4-2','Q4-3')}

def call(script,*args):
    subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,check=True)

def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')

def verify(deep=False):
    call('verify_results.py')
    call('test_time_rules.py')
    call('audits/finalize_df.py')
    if deep:call('audits/logic_audit.py')

def package():
    # No copies are certified until the original artifacts pass verification.
    verify()
    frozen=json.loads((ROOT/'results/frozen_numbers.json').read_text(encoding='utf-8'))
    by_problem={s['problem']:s for s in frozen}
    if len(frozen)!=4 or set(by_problem)!=set(PROBLEMS):raise ValueError('Incomplete frozen baseline')
    for p in PROBLEMS:
        actual=json.loads((ROOT/'results'/f'{p}_summary.json').read_text(encoding='utf-8'))
        if by_problem[p]!=actual:raise ValueError(f'Frozen/source mismatch: {p}')
    shutil.copy2(ROOT/'results/frozen_numbers.json',ROOT/'frozen_numbers.json')
    for folder,problems in GROUPS.items():
        dest=ROOT/folder
        (dest/'结果').mkdir(parents=True,exist_ok=True)
        for p in problems:
            for suffix in ('daily.csv','physical_10min.csv','plans.npz','summary.json'):
                shutil.copy2(ROOT/'results'/f'{p}_{suffix}',dest/'结果'/f'{p}_{suffix}')
        write(dest/'README.md',f'# {folder}\n\n共享源码为`../run_year.py`与`../opt/opt_core.py`；本目录不复制计算内核。\n\n'
            '运行`一键全年计算.cmd`会重新计算本问题并覆盖主结果，随后重新核验、打包。\n'
            '`结果/`是已核验基线的交付副本，不能在此手工修改后当作新冻结结果。\n'
            '十分钟全轨迹包含365天；题目正式输出期为2月1日—12月31日334天。\n'
            '出图表格统一见`../出图数据/表格/`。本目录未生成官方result*.xlsx或正式论文。\n')
        cli='Q4' if len(problems)==2 else problems[0]
        write(dest/'一键全年计算.cmd',f'@echo off\npython "%~dp0..\\project.py" run --problem {cli} %*\nif errorlevel 1 exit /b 1\npython "%~dp0..\\project.py" package\nexit /b %errorlevel%\n')
        sections=[f'# {folder}方法说明','', '本文件描述当前实现，不借用一次优化版的数字或完成声明。','',
            '## 共同时间与物理规则','',
            '输入右端点00:10对应00:00—00:10。00:00新计划覆盖00:10—次日00:10；自然日拼接前一日末段与当日前143段。',
            '初始SOC6000 kWh，年初首段承诺0；1月预运行连续衔接2月。容量1200—10800 kWh，充放电各5000 kW，效率各0.9。',
            '不设置日终SOC等式；年终实际SOC6000。计划层尚未协调年终目标，由最后一天执行保护实现。','']
        for p in problems:
            staged=p in ('Q3','Q4-3')
            sections.extend([f'## {p}','',
                ('00/06/12/18四次发布，调整仅覆盖尚未执行的计划段；目标按历史最小承诺和相邻修订费用的真实链式增量构建MILP。执行为十分钟贪心反馈。'
                 if staged else '00:00一次计划，线性规划确定购电承诺；十分钟储能反馈执行使用最多36段预测窗口，只执行第一步。首段及日末窗口会截短。'),
                ('电价使用附件4历史产生的因果预测，实际电价只用于结算。' if p.startswith('Q4') else '电价使用附件1已知固定日内曲线。'),
                ('允许附件1、2、3'+('、4。' if p.startswith('Q4') else '。') if staged else '负载与光伏预测使用附件1、2。'+('另使用附件4预测电价。' if p.startswith('Q4') else '通用预处理仍依赖附件3、4存在，但不参与Q2决策。')),
                '风险R取此前已结束日的累计有符号净负荷误差0.80分位数，历史不足20天取0，内核上限4800 kWh。',
                '光伏零值掩码是过去30天经验支持窗口，不是严格天文夜间判据。当前段实际值用于反馈执行，需明确区间内实时观测假设。',''])
        write(dest/'方法说明.md','\n'.join(sections))
        evaluation=[f'# {folder}模型评价','', '数值来自本版冻结基线；D为同年事后敏感性对照，不用评价期最优点自动改参。','',
            '|模型|365天实际费用/元|334天实际费用/元|实际年终SOC/kWh|','|---|---:|---:|---:|']
        for p in problems:
            s=by_problem[p]
            evaluation.append(f"|{p}|{s['calendar_year_cost_yuan']:.2f}|{s['formal_period_cost_yuan']:.2f}|{s['final_soc_kwh']:.2f}|")
        evaluation.extend(['','## 已有证据与限制','',
            '物理平衡、SOC连续性、映射和费用恒等式核验通过；D九点扫描与F全知对照已完成。',
            '尚不能称全年全局最优。年终保护可能增加费用；当前段反馈存在信息假设；R覆盖率不是无应急概率，F不是纯因果贡献分解。',
            '具体影响、试验范围及数值误差适配见`../audits/逻辑检查报告.md`。',
            '未完成正式论文、官方Excel模板导出、Bootstrap及新增消融实验，不能照搬旧包的封版声明。',''])
        write(dest/f'模型评价_{folder[:3]}.md','\n'.join(evaluation))
    summary=['# 最终基线结果汇总','', '以`results/`为主结果、根目录`frozen_numbers.json`为核验后的数字副本。无储能残值校正。','',
        '|模型|365天/万元|正式期334天/万元|年终SOC/kWh|','|---|---:|---:|---:|']
    for p in PROBLEMS:
        s=by_problem[p];summary.append(f"|{p}|{s['calendar_year_cost_yuan']/10000:.4f}|{s['formal_period_cost_yuan']/10000:.4f}|{s['final_soc_kwh']:.2f}|")
    summary.extend(['','D/F扫描结果与限制见`audits/DF补算汇总.md`及`audits/逻辑检查报告.md`。',''])
    write(ROOT/'最终结果汇总.md','\n'.join(summary))
    write(ROOT/'答案总结/问题234结果说明.md','\n'.join(summary)+
        '\n各问题方法与评价见三个“最终模型”目录。正式期费用按自然日00:00—24:00结算，计划行则跨至次日00:10。\n'
        '所有D/F点均是本版计算，不引用其他版本；扫描q=0.65的改善不自动改变基线q=0.80。\n')
    write(ROOT/'封版核验.md','PASSED_WITH_WARNINGS\n\n# 交付结构核验\n\n'
        '分问题结果副本、冻结数字、时间回归、物理与费用恒等式、D/F完成状态通过核验。\n'
        '本结论只认证当前结果与结构一致，不是最终竞赛论文G6批准。\n\n'
        '未闭合：年终目标与计划协调、实时反馈信息假设、R/F解释、附件定位可移植性。\n'
        '未交付：正式论文及官方result*.xlsx，不冒充完整提交包。\n'
        '详情见`audits/逻辑检查报告.md`。运行`python project.py package`重新核验并刷新交付副本。\n')
    # Record every delivery artifact except regenerating inventories / transient caches.
    excluded={'最终文件清单.json','annual_audit.json'}
    inventory=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name in excluded:continue
        inventory.append(dict(file=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write(ROOT/'最终文件清单.json',json.dumps(dict(status='PASSED_WITH_WARNINGS',
        result_authority='results/',frozen_numbers='frozen_numbers.json',files=inventory),ensure_ascii=False,indent=2))
    print('Verified and packaged all four models; baseline algorithm and fees unchanged.')

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('--problem',choices=(*PROBLEMS,'Q4'))
    smoke=sub.add_parser('smoke');smoke.add_argument('--problem',choices=PROBLEMS,default='Q2')
    smoke.add_argument('--days',type=int,default=2)
    check=sub.add_parser('verify');check.add_argument('--deep',action='store_true')
    sub.add_parser('package');sub.add_parser('export')
    args=ap.parse_args()
    if args.command=='run':
        for p in (('Q4-2','Q4-3') if args.problem=='Q4' else ((args.problem,) if args.problem else PROBLEMS)):
            call('run_year.py','--problem',p)
    elif args.command=='smoke':
        if not 1<=args.days<=365:ap.error('--days must be between 1 and 365')
        import run_year as model
        # Avoid run_year --days silently replacing the delivered annual CSVs.
        original=model.OUT
        try:
            model.OUT=Path(tempfile.mkdtemp(prefix='codex-model-smoke-'))
            model.simulate(model.prepare(),args.problem,args.days)
            print('Isolated smoke artifacts:',model.OUT)
        finally:model.OUT=original
    elif args.command=='verify':verify(args.deep)
    elif args.command=='package':package()
    elif args.command=='export':
        call('出图数据/export_plot_data.py')
        call('audits/finalize_df.py')

if __name__=='__main__':main()
