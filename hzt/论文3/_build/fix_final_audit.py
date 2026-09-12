"""修正最终核查内容并备份原交付文件；不改模型代码或结果数据。"""
from pathlib import Path
from datetime import datetime
import shutil

base = Path(__file__).resolve().parent.parent
backup = base / '_build' / 'backups' / datetime.now().strftime('%Y%m%d_%H%M%S')
for rel in ['论文3.docx', '论文3_LaTeX/论文3.tex', '论文3_LaTeX/论文3.pdf', '_build/src/01_前置.md', '_build/src/02_模型一问题二.md', '_build/src/03_问题三四检验评价.md']:
    source = base / rel
    target = backup / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

def paragraph(text, start, replacement):
    lines = text.splitlines()
    found = [i for i, line in enumerate(lines) if line.startswith(start)]
    assert len(found) == 1, (start, found)
    lines[found[0]] = replacement
    return '\n'.join(lines) + '\n'

p = base / '_build/src/01_前置.md'
s = p.read_text(encoding='utf-8')
old = '风险层的安全储备与价格无关（临界比中价格被约去），故两个子问题共用同一储备水平，四个（子）问题结果可直接横向比较。'
assert old in s
s = s.replace(old, '风险层统一采用事前分位数 0.80，但储备量由各自预测残差决定：问题 4-2 沿用问题二的储备规则，问题 4-3 沿用问题三各发布时点的储备规则。统一分位数并不意味着两个子问题的储备量相同；横向费用比较同时涉及预测、决策与执行配置的差异。')
p.write_text(s, encoding='utf-8')

p = base / '_build/src/02_模型一问题二.md'
s = p.read_text(encoding='utf-8')
for a, b in [('2,882.475', '480.4124'), ('2,672.59', '445.4317'), ('3,191.364', '531.8940')]:
    assert s.count(a) == 1
    s = s.replace(a, b)
s = paragraph(s, '@@P 即实际净负荷减预测净负荷；', r'@@P 即实际净负荷减预测净负荷；$\xi_{j,t}>0$ 表示实际净负荷高于预测，计划量偏少。储备统计采用自然日内已经完成的新计划可执行窗口：排除已由上一日承诺锁定的 00:00—00:10 段，只累计当日 00:10—24:00 的 143 段残差，不把计划窗口中的次日首段并入当天样本。第 $d$ 天只使用 $j<d$ 的样本，取累计残差电量的 80% 分位数：')
s = s.replace(r'\sum_{t=1}^{T}\xi_{j,t}\Delta t', r'\sum_{t=2}^{T}\xi_{j,t}\Delta t')
s = paragraph(s, '@@P 储备进入决策前须做两项处理。', r'@@P 历史累计残差样本不足 20 天时，取 $R_d=0$；达到 20 天后按式(13)计算。储备进入决策前须做两项处理。其一是非负截断：分位数本身可能为负，此时取 $R_d=\max(R_d,0)$。其二是内核有效上限：内核实际采用的储备不超过 $(E_{\max}-E_{\min})/2=4800$ kWh，记作 $\widehat{R}_d=\min(R_d,4800)$。储备下界按充电能力从计划窗口初始状态爬坡：令 $t_{lo}=\lceil\max(0,E_{\min}+\widehat{R}_d-E^{plan}_{d,0})/(\eta_c P_{st}^{\max}\Delta t)\rceil$，计划段采用从零开始的局部序号，零号段结束状态为 $E^{plan}_{d,1}$；局部序号小于 $t_{lo}$ 时只要求储电量不低于 $E_{\min}$，自局部序号 $t_{lo}$ 起要求不低于 $E_{\min}+\widehat{R}_d$。原始储备、有效储备与实际储电量是不同的量。以式(18)从 1 开始的状态序号表示，约束为')
s = s.replace(r'\quad(t\ge t_{lo})', r'\quad(t\ge t_{lo}+1)')
s = paragraph(s, '@@P 日前购电与储能调度：', r'@@P 日前购电与储能调度：0:00 时的实际储电量记为 $E^{act}_{d,0}$，此时当日 00:00—00:10 段已由上一日尾段承诺锁定。先用该承诺及当日首段的负载、光伏预测，按充放电效率、功率和容量边界预测首段结束状态，记为 $E^{plan}_{d,0}$；再以该预测状态求解从当日 00:10 至次日 00:10 的 144 段计划。式(15)—(19)中的状态均为计划状态，$E^{plan}_{d,T}$ 是该计划窗口末端状态，区别于自然日 24:00 的实际状态。窗口内预测、电价与购电变量均按计划轴配对，次日首段采用当日首段负载和电价预测的延续值，光伏预测取零。日前目标为：')
for a,b in [(r'E_{d,t-1}', r'E^{plan}_{d,t-1}'), (r'E_{d,t}', r'E^{plan}_{d,t}'), (r'E_{d,T}', r'E^{plan}_{d,T}'), (r'E_{d,0}', r'E^{plan}_{d,0}')]:
    s = s.replace(a,b)
s = s.replace('第二项是日末储电量的水值', '第二项是计划窗口末端储电量的水值')
s = paragraph(s, '@@P 逐日因果：', r'@@P 逐日因果：整体求解按逐日前向推进。第 $d$ 天先用 $j<d$ 的历史数据完成预测与储备计算，再由午夜实际状态 $E^{act}_{d,0}$、已锁定首段承诺及首段预测得到 $E^{plan}_{d,0}$，以该预测状态求解式(16)—(19)。实际执行时，先执行当日 00:00—00:10 的上一日尾段承诺，再按映射执行本日新计划；自然日 24:00 的实际储电量作为次日午夜状态连续递推。当天费用按物理轴配对后的有效承诺及紧急购电量结算。日末只把已经发生的当日实际值与残差并入历史库，尚未发生的次日首段不进入该日残差样本。任意一天的计划均不引用该天及以后的实际数据。')
s = s.replace('最小化窗口内的预计紧急购电费用与日末电量水值之和', '最小化窗口内的预计紧急购电费用减去窗口末端储电量水值')
s = s.replace('全天累计有符号误差的 0.80 分位数', '当日 00:10—24:00 累计有符号误差的 0.80 分位数')
p.write_text(s, encoding='utf-8')

p = base / '_build/src/03_问题三四检验评价.md'
s = p.read_text(encoding='utf-8')
s = paragraph(s, '@@P 储备量的统计口径也随之分层。', r'@@P 储备统计窗口与计划窗口须分别定义。$J_k$ 包含次日 00:00—00:10 段，但当天日末尚未观察到该段真实值；代码只累计当日已经完成的物理段残差。记物理残差窗口为 $I_0=\{2,\ldots,144\}$、$I_6=\{37,\ldots,144\}$、$I_{12}=\{73,\ldots,144\}$、$I_{18}=\{109,\ldots,144\}$。对发布时刻 $k$，使用该时刻自身预测版本在 $I_k$ 内的历史残差计算储备：')
s = paragraph(s, '@@EQ R_d^{(k)}=', r'@@EQ R_d^{(k)}=Q_{0.80}\left(\left\{\sum_{i\in I_k}\xi_{\ell,i}^{(k)}\Delta t:\ \ell<d\right\}\right) | (24)')
s = paragraph(s, r'@@P 式中 $\xi_{j,t}$', r'@@P 式中 $\xi_{\ell,i}^{(k)}$ 是第 $\ell$ 日发布时刻 $k$ 的预测版本在物理段 $i$ 上的带符号净负荷残差。四个物理残差窗口分别为 143、108、72、36 段，区别于 144、109、73、37 段的计划窗口。历史样本不足 20 天时储备取零，之后按式(24)取分位数，并沿用非负截断、4800 kWh 有效上限和充电能力爬坡规则。')
s = s.replace('3.1 节假设 6 声明的口径', '第三章假设（5）声明的口径')
s = paragraph(s, '@@P 需要注意，$c_t[', r'@@P 令 $d_t=(m_t-z_t)^+$，则 $c_t[\min(m_t,z_t)-m_t]=-c_t d_t$ 严格相等，不存在额外的常数差。代码用于优化的目标基准项为 $c_t\min(m_t,z_t)=c_t m_t-c_t d_t$；它比式(25)的基准增量高一个与 $z_t$ 无关的常数 $c_t m_t\Delta t$。因此两种目标得到相同的最优解，但若比较目标值与费用增量，须扣除该常数。最终费用由完整承诺路径另行结算，不直接采用求解器的目标值。')
s = paragraph(s, '@@P 每个发布时刻的决策只覆盖', r'@@P 0:00 首次计划尚无上一版承诺和历史最低值，使用式(16)的正常购电费目标，以预测执行已锁定首段后的储电量为起点求解线性规划。6:00、12:00、18:00 修订则以当时实际储电量为起点，只覆盖各自剩余计划窗口，求解如下单阶段问题：')
s = s.replace('由此 $-c_t d_t$ 恰好等于 $c_t[\min(m_t,z_t)-m_t]$（相差常数 $-c_t m_t$），凹项被线性化。', '由于目标中 $d_t$ 的系数为负，最小化会在可行分支内把 $d_t$ 推至上界，从而在最优解处实现 $d_t=(m_t-z_t)^+$。因此 $-c_t d_t=c_t[\min(m_t,z_t)-m_t]$，凹项被精确线性化；代码基准费目标再加回与决策无关的 $c_t m_t$。')
s = s.replace('下调与上调的费用之比为 4.05，电量之比为 1.47', '上调与下调的费用之比为 4.05，电量之比为 1.47')
s = s.replace('四个发布时点的剩余窗口长度不同（144、109、73、37 个时段）', '四个发布时点的残差统计窗口长度不同（143、108、72、36 个时段）')
p.write_text(s, encoding='utf-8')
print('backup:', backup)
print('final audit content corrected')
