# -*- coding: utf-8 -*-
"""
regression_test.py — 本轮修复的轻量回归测试
================================================
只覆盖“曾经真出过问题、且修好后容易再次退化”的契约，不做全链路性能测试：

    1. ChiK 长度契约：k/chi 等长；background/mu_norm 在能量网格
    2. theory_chi 的 nleg 过滤：MS（nleg>4）不得混入指纹理论谱
    3. indistinguishable 不包含最优候选自身
    4. WT 矩阵方向：(len(k_centers), len(r))
    5. strict 门禁：Fe 箔平台 0.79 → strict 拒绝 / --no-strict 降级
    6. S7 motif 路径：lepidocrocite → Fe–O/Fe–Fe 多壳层 CN
    7. S7 L1 路径：goethite（库外）→ 仅第一壳，且不冒充全结构
    8. 磁铁矿多位点：magnetite_mixed 联合模型 → confident 全覆盖
    9. mixture 壳层选择：Fe-O@1.89(⅓×4) 与 Fe-O@2.06(⅔×6) 不合并
    10. vary_s02 合成回路：N 固定、放开 S₀² 能回收真值（标定前提）
    11. L1 配体感知选峰：Pt–Cl 不被低 R 白线伪峰带偏（ligand=Cl）
    12. L1 长键窗口：Pt–Pt 2.77 不被 rmax=2.3 截断（曾 Rf=1.0 伪失败）
    13. mixture 相分数回收：磁铁矿 tet/oct 第一壳 0.25/0.75 合成谱
    14. 荧光自吸收门禁：Δμ≥1.0 拒绝 CN，透射模式不误伤
    15. Booth 自吸收反解：Eq.2 前向 → Eq.4 回收（薄/厚极限）
    16. Booth 全链合成：真实 μ → 前向荧光谱 → 归一化/autobk → 校正
    17. WT 定位锚定 FT 峰：自吸收校正前后不得在两壳分支间跳
    18. σ² 共享组：多路径共享一个 σ² 变量（Artemis 式约束）
    19. CN 系统误差预算随拟合条件动态合成（不再恒为 ±30%）
    20. HTML 报告：图片内联、强相关/不可靠项标红
    21. 配置文件未知扩展名给出干净 ValueError（p.suffix 回归）
    22. Hanning 窗契约：端点 0、平台 1，供浏览器端 FT 与产线对齐
    23. WT sweep：9 组 (σ,kweight) 全部锚定 FT 第一峰 ±0.5 Å，CSV/图可写出
    24. 交互式窗口 HTML：内嵌 χ(k)、滑块齐全、推荐窗口落在 r1±0.15 Å
    25. 多 k 权重联合拟合：三分支 Rf、第一壳键长稳定、N_idp 不变
    26. Back-FT：q 轴单调、包络非负、Re χ(q) 与 k²χ 同相
    27. NiO 共线 MS：nleg>2 选入、自由变量数不变、ΔR 按 reff_ratio 缩放
    28. XANES flattened 平台≈1；合成颈部双台阶识别 + E0 跳变风险
    29. LCF 三端元权重回收；PCA 前 3 主成分累计解释 >99.9%
    30. 死时间倍乘/分数两条公式；8 点宽 glitch 的 despike vs deglitch
    31. WT 壳层切片：Fe–Fe 的 k_peak 高于 Fe–O ≥0.8 Å⁻¹；模型 WT 残差 R<0.15
    32. ε_R / χ²_ν 为正；双壳相对单壳的 F-test 置信度 >0.95
    33. C3 合成回收：C3 真值 ±5e-5；N_idp 不足时自动降级不崩溃

Phase II（B5–B8）新增 9 项契约：
    - 复数小波：反相 χ 的 complex_r_factor=4.0，幅值残差=0.0
    - 元素理论 k_peak 标尺：O < Cl < Fe 且随 σ² 左移
    - 多扫描合并：MAD 离群剔除、σ_μ 传播、默认均值逐位兼容
    - XANES 荧光自吸收：Tröger 反解恢复白线/LCF，分母防爆门禁
    - 边前 Pseudo-Voigt：双峰质心误差 <0.15 eV，指标表同步
    - 各向同性晶格约束：M 个 ΔR → 1 个 α；实验 NiO 诚实降级
    - 部分占据 CIF：共占位合并、占据度缩放简并度（6×0.75=4.5）
    - C4 回收 + 非高斯 g(R)：面积=N、正偏态尾长判据
    - 多数据集联合拟合：5 个全局变量、变温爱因斯坦 σ²(T) 回收

Phase III（B9–B12）新增 15 项契约：
    - 17 键配置文件生效；C4 的 N_idp 降级与 g(R) 导出
    - σ_μ 落盘/SNR 门禁；WT 元素标尺自动挂载与 L3 白线面积
    - .prj round-trip 与 MCA 坏道剔除；MBACK 短窗口归一化
    - AUTOBK chi_std/clamp；SPOIL + MCR-ALS 盲解谱
    - 相移校正 FT/WT；异核共占位双分支；关联 Debye；BVS 软约束
    - R/q/WT 多空间拟合；双吸收边 ΔE₀ 分组 + AIC/BIC；SAC/DAC 团簇

P4+ 新增：
    - T7 多元素综合报告：自动分元素、并列结果、不推荐 Top-1
    - T8 XANES–EXAFS 软校验：只产生警告，不改变 cn_reliable
    - P5-A T5 标定产物完整性 / --only 增量合并 / --s02-from-t5 溯源
    - P5-B Mo 双壳 bcc 标定 / Ce σ² 压界与归一化伪迹拒绝
    - P5-C 固定 COD ID 补齐 18 个常见相 / coverage 无缺口
    - P5-prep 标准谱源隔离池 / 5 个未通过元素门禁冻结

当前共 135 项契约回归（B1-1 至 B20-1，外加 P0a–P4 正式 CN 契约、P4+
CIF 抓取/条件 CN/T7/T8/P5-A/P5-B/P5-C/P5-prep 契约，以及 Fe 化学态
S₀² 多态契约）。

运行:
    python regression_test.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import xafs  # noqa: E402,F401

from tests.test_io_config import IOConfigTests  # noqa: E402
from tests.test_preprocess import PreprocessTests  # noqa: E402
from tests.test_motifs_cif import MotifsCIFTests  # noqa: E402
from tests.test_fitting_cn import FittingCNTests  # noqa: E402
from tests.test_pipeline_report import PipelineReportTests  # noqa: E402


class RegressionTest(IOConfigTests, PreprocessTests, MotifsCIFTests, FittingCNTests, PipelineReportTests, unittest.TestCase):
    """聚合入口：全部测试方法由 tests/ 下的 mixin 提供，共 199 项。"""


if __name__ == "__main__":
    # unittest 默认写 stderr；PowerShell 会把任何 stderr 记录当成
    # NativeCommandError，导致测试通过也显示“失败”。统一写 stdout。
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RegressionTest)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
