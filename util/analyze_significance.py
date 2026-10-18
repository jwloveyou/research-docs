#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_significance.py — 统计显著性 / 效应量 / Bootstrap 置信区间分析（任务#2 缺口）

对最终融合模型的全部「有/无」组件消融，做 3-seed 配对显著性分析：
  - 配对均值差 Δ、样本标准差 s_Δ
  - 配对 t 检验（df = n-1 = 2，诚实报告 n=3 的局限）
  - Cohen's d 效应量（d = mean(Δ)/s_Δ，配对版）
  - Bootstrap 95% 置信区间（配对重采样，B=100000）

所有原始 3-seed 值均从训练日志逐项核对（来源见每项 source 字段），
不使用任何纸面值。n=3 时 t 检验与 bootstrap 均偏弱，故同时报告
「零交叉」判据（min(A) >= max(B) 表示三 seed 无任何重叠，方向稳健）。

用法：python util/analyze_significance.py
输出：Markdown 表格（可直接粘入 研究总报告.md §2.1 / §3）
"""
import math
import random

# ---------------------------------------------------------------
# 原始数据（3-seed，seed 42/43/44，从日志逐项核对）
# ---------------------------------------------------------------
# 每项: (名称, A组名, A值列表, B组名, B值列表, 指标, 来源)
COMPARISONS = [
    # 组件① 噪声增广 on/off (gowalla, R@20)
    dict(
        name="组件① 噪声增广 (gowalla R@20)",
        a_name="有噪声 (fusion)", a=[0.1880, 0.1884, 0.1883],
        b_name="无噪声 (LightGCN)", b=[0.1763, 0.1763, 0.1761],
        metric="R@20", source="R24 embed256 3seed / R28 '无'侧 3seed (ssl_reg=0, embed256, 100ep)",
    ),
    # 组件① 噪声增广 on/off (gowalla, NDCG)
    dict(
        name="组件① 噪声增广 (gowalla NDCG@20)",
        a_name="有噪声 (fusion)", a=[0.1598, 0.1598, 0.1596],
        b_name="无噪声 (LightGCN)", b=[0.1508, 0.1508, 0.1506],
        metric="NDCG@20", source="同上",
    ),
    # 组件② win30 退火 on/off (gowalla, R@20)
    dict(
        name="组件② win30 退火 (gowalla R@20)",
        a_name="有退火 (win30)", a=[0.1880, 0.1884, 0.1883],
        b_name="无退火 (固定 λ)", b=[0.1834, 0.1834, 0.1829],
        metric="R@20", source="R24 embed256 3seed (有) / R28 ablation 3seed (无, ssl_anneal=0, embed256, 100ep)",
    ),
    # 组件② win30 退火 on/off (gowalla, NDCG)
    dict(
        name="组件② win30 退火 (gowalla NDCG@20)",
        a_name="有退火 (win30)", a=[0.1598, 0.1598, 0.1596],
        b_name="无退火 (固定 λ)", b=[0.1527, 0.1525, 0.1520],
        metric="NDCG@20", source="同上",
    ),
    # 组件③ 容量缩放 128->256 (gowalla, R@20)
    dict(
        name="组件③ 容量缩放 128→256 (gowalla R@20)",
        a_name="embed256", a=[0.1880, 0.1884, 0.1883],
        b_name="embed128", b=[0.1861, 0.1854, 0.1859],
        metric="R@20", source="R24 embed256 3seed / R18-R20 embed128 3seed",
    ),
    # 组件③ 容量缩放 128->256 (gowalla, NDCG)
    dict(
        name="组件③ 容量缩放 128→256 (gowalla NDCG@20)",
        a_name="embed256", a=[0.1598, 0.1598, 0.1596],
        b_name="embed128", b=[0.1578, 0.1577, 0.1580],
        metric="NDCG@20", source="同上",
    ),
    # 组件④ 冻结语义注入 sem0/sem1 (lastfm-2k, R@20)
    dict(
        name="组件④ 冻结语义注入 (lastfm-2k R@20)",
        a_name="sem_emb=1", a=[0.4209, 0.4241, 0.4201],
        b_name="sem_emb=0", b=[0.4183, 0.4201, 0.4164],
        metric="R@20", source="R53 lastfm 3seed",
    ),
    # 组件④ 冻结语义注入 sem0/sem1 (lastfm-2k, NDCG)
    dict(
        name="组件④ 冻结语义注入 (lastfm-2k NDCG@20)",
        a_name="sem_emb=1", a=[0.3683, 0.3688, 0.3674],
        b_name="sem_emb=0", b=[0.3658, 0.3658, 0.3649],
        metric="NDCG@20", source="R53 lastfm 3seed",
    ),
]


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_bootstrap_ci(a, b, B=100000, alpha=0.05, seed=42):
    """配对 bootstrap：对 seed 索引重采样，估计 mean(a-b) 的分布。"""
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(mean([diffs[i] for i in idx]))
    boots.sort()
    lo = boots[int(B * alpha / 2)]
    hi = boots[int(B * (1 - alpha / 2))]
    return lo, hi


def paired_t(a, b):
    """配对 t 检验（双尾）。返回 (t, df, p)。df=n-1。"""
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    m = mean(diffs)
    if n < 2:
        return None
    s = sd(diffs)
    if s < 1e-12:
        return (float("inf"), n - 1, 0.0)
    t = m / (s / math.sqrt(n))
    df = n - 1
    # 用精确学生 t 分布的 CDF（正则化不完全 beta），避免 scipy 依赖
    p = 2 * (1 - _student_t_cdf(abs(t), df))
    return t, df, p


def _student_t_cdf(t, df):
    """正则化不完全 beta 计算学生 t 分布 CDF。"""
    if df <= 0:
        return 0.5
    x = df / (df + t * t)
    ib = _betainc(0.5 * df, 0.5, x)
    if t >= 0:
        return 0.5 + 0.5 * (1 - ib)  # P(T<=t)
    return 0.5 - 0.5 * (1 - ib)


def _betainc(a, b, x):
    """正则化不完全 beta I_x(a,b)，连分数法（数值稳定）。"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    # 连分数（Lentz 法）
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return math.exp(lbeta) * h / a


def main():
    print("# 3-seed 配对显著性分析（融合模型组件「有/无」消融）\n")
    print("> 诚实说明：n=3 时配对 t 检验（df=2）与 bootstrap CI 统计功效有限；")
    print("> 故同时报告 **零交叉** 判据（min(A) >= max(B) 表示 3 seed 方向完全一致、无任何重叠），")
    print("> 该判据对 n=3 最稳健。所有原始值均从训练日志逐项核对（非纸面值）。\n")
    print("| 对比（指标） | A组（有） | B组（无） | Δ 均值±s | 配对 t (df=2) | p 值 | Cohen's d | Bootstrap 95% CI | 零交叉 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in COMPARISONS:
        a, b = c["a"], c["b"]
        diffs = [a[i] - b[i] for i in range(len(a))]
        dm = mean(diffs)
        ds = sd(diffs)
        t, df, p = paired_t(a, b)
        d_effect = dm / ds if ds > 1e-12 else float("inf")
        lo, hi = paired_bootstrap_ci(a, b)
        zero_cross = "✅ 是" if min(a) >= max(b) else "❌ 否"
        tstr = "∞" if (t == float("inf") or t > 1000) else f"{t:.2f}"
        pstr = f"{p:.4f}" if p > 0 else "<0.0001"
        dstr = "∞" if d_effect == float("inf") else f"{d_effect:.2f}"
        print(f"| {c['name']} | {c['a_name']}={mean(a):.4f} | {c['b_name']}={mean(b):.4f} "
              f"| {dm:+.4f}±{ds:.4f} | {tstr} | {pstr} | {dstr} | [{lo:+.4f}, {hi:+.4f}] | {zero_cross} |")
        print(f"| 　（3-seed 明细） | {a} | {b} | — | — | — | — | — | — |")
    print()
    print("来源逐项：")
    for c in COMPARISONS:
        print(f"- {c['name']}: {c['source']}")
    print()
    print("结论：组件①（噪声，gowalla +6.7%）、②（win30 退火，gowalla +2.7%/+4.8%）、③（容量 128→256，gowalla +1.3%）、")
    print("组件④（语义，lastfm +0.82%）四者 3-seed 全部零交叉 + 配对 t 显著 + bootstrap CI 不含 0，")
    print("增益方向稳健、非 seed 偶然。组件②「无」侧 3-seed 已于 R72 收齐（r28 ablation gowalla_noanneal s42/43/44），")
    print("与「有」侧（R24 embed256 3seed）逐项匹配（仅 ssl_anneal 2→0），完整消融表闭合。")


if __name__ == "__main__":
    main()
