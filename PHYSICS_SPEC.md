# PHYSICS_SPEC.md — 核心物理模型与数学公式白盒规范

> **使用说明**：本文件是 `xafs-auto-pipeline` 项目所有物理公式、符号约定、正负号方向与单位制的唯一权威标准（Single Source of Truth）。凡在核心计算模块实现下述公式的函数，须在 Docstring 开头标注 `[Implements: EQ-XX]` 双向锚点，确保与本规范及 [AGENTS.md](./AGENTS.md) 100% 一致。

---

## 1. 全局基础单位制约定

除非在特定条目中另有显式说明，本项目内存数据结构（[xafs_core.py](./xafs_core.py)）内部统一采用以下物理单位：

- **能量（Energy, $E, E_0$）**：电子伏特（$\text{eV}$）
- **波矢（Wavevector, $k$）**：埃的倒数（$\text{\AA}^{-1}$）
- **实空间距离 / 键长 / 半程路径（Distance, $R, R_{\text{eff}}$）**：埃（$\text{\AA}$）
- **纳米颗粒输入口径（`diameter_nm`）**：纳米（$\text{nm}$）；进入球截断无量纲比值前强制换算为埃（$d_{\text{Å}} = 10 \cdot d_{\text{nm}}$）
- **德拜-瓦勒因子（Debye-Waller factor, $\sigma^2$）**：平方埃（$\text{\AA}^2$）
- **三阶 / 四阶累积量（$C_3, C_4$）**：$\text{\AA}^3$ / $\text{\AA}^4$
- **角度与相位（Angle / Phase, $\phi$）**：角度用度（$^\circ$），散射相位用弧度（$\text{rad}$）

---

## 2. 核心公式定义与实现锚点

### `[EQ-01]` 光电子波矢转换公式
- **数学表达式**：
  $$k = \sqrt{\frac{2m_e}{\hbar^2}(E - E_0)} = \sqrt{K_2 \cdot (E - E_0)}, \quad K_2 = 0.2624685\text{ \AA}^{-2}\text{eV}^{-1}$$
- **边界与奇异点约定**：
  - 仅在 $E > E_0$ 区域定义正实波矢；$E \le E_0$ 处截断返回 $k = 0$，严禁产生复数或 `NaN`。
- **代码实现锚点**：
  - `src/xafs/domain/xafs_core.py :: K2`
  - `src/xafs/adapters/athena_bkg_ft.py :: wave_number`

---

### `[EQ-02]` 标准 EXAFS 振荡方程（多路径求和）
- **数学表达式**：
  $$\chi(k) = \sum_{j=1}^{N_{\text{paths}}} \frac{N_j S_0^2 F_j(k)}{k R_j^2} \exp\left(-2k^2\sigma_j^2\right) \exp\left(-\frac{2R_j}{\lambda(k)}\right) \sin\left(2kR_j + \phi_j(k)\right)$$
  其中有效键长 $R_j = R_{\text{eff}, j} + \Delta R_j$。
- **歧义消除约定**：
  - $S_0^2$ 与配位数 $N_j$ 独立定义，禁止在未标定或未经门禁确认时静默合并；
  - 德拜-瓦勒衰减指数为 $-2k^2\sigma_j^2$；相移相位符号为 $+\sin(2kR_j + \phi_j(k))$。
- **代码实现锚点**：
  - `src/xafs/adapters/artemis_stats.py :: exafs_model_k`

---

### `[EQ-03]` 正向傅里叶变换（$k \rightarrow R$ 空间）与 Hanning 窗约定
- **数学表达式**：
  $$\tilde{\chi}(R) = \mathcal{F}\!\left\{ k^w \chi(k) \cdot W_{\text{Hanning}}(k) \cdot e^{-i\phi_{\text{tot}}(k)} \right\}_{e^{-i 2kR}}$$
  具体归一化因子、均匀 $k$ 网格重采样（$\Delta k_{\text{grid}} = 0.05\text{ \AA}^{-1}$）与窗端点与 `larch.xafs.xftf` 严格一致。
- **歧义消除约定**：
  - 相位因子必须为 $e^{-i 2kR}$（含因子 $2$），严禁误写为 $e^{-i 2\pi k R}$；
  - 仅在显式提供 `phase_correct = φ_tot(k)` 时做相移校正，否则默认返回未修正的表观距离（表观 $|\chi(R)|$ 峰位比真实键长系统偏短约 $0.3\text{–}0.5\text{ \AA}$），报表严禁混淆表观峰位与拟合真实键长 $R$。
- **代码实现锚点**：
  - `src/xafs/adapters/athena_bkg_ft.py :: ft_to_r_space`

---

### `[EQ-04]` Stern / IFEFFIT 独立信息点数与过拟合门禁
- **数学表达式**：
  $$N_{\text{idp}} = \frac{2 (k_{\max} - k_{\min}) (R_{\max} - R_{\min})}{\pi}$$
- **硬性统计约束（严守 `AGENTS.md` 红线 3）**：
  - 严格采用 IFEFFIT / larch / Artemis 标准口径，**严禁额外 $+2$**；
  - 硬过拟合红线：$N_{\text{var}} < N_{\text{idp}}$（若 $N_{\text{var}} \ge N_{\text{idp}}$，必须拒绝/截断高阶壳层）；
  - 自由度冗余建议：$N_{\text{var}} \le \frac{2}{3} N_{\text{idp}}$（超出给出 Warning）；
  - 网格过采样误差棒修正因子：`nidp_scale_factor` $= \sqrt{\max(N_{\text{grid}} - N_{\text{var}}, 1) / \max(N_{\text{idp}} - N_{\text{var}}, 1)}$，独立导出至 `N_err_nidp` 与 `R_err_nidp`（不篡改 CSV 基线）。
- **代码实现锚点**：
  - `src/xafs/adapters/artemis_stats.py :: independent_points`
  - `src/xafs/domain/xafs_core.py :: FitResult.overfit`
  - `src/xafs/core/fit_shell_plan.py :: select_shell_paths`

---

### `[EQ-05]` 荧光自吸收 Booth 反解与透射光学厚度门禁
- **物理约定**：
  - **荧光模式（Booth & Bridges, *Phys. Scr.* 2005）**：未校正且 $\Delta\mu \ge 1.0$ 时，强制触发 `cn_gate=True`、`cn_reliable=False`；显式给定化学式/厚度/密度/角度时在 $k$ 空间仅修正振幅，不改相位与 $\mu(E)$；
  - **透射模式**：不受荧光 `cn_gate` 拦截，但按边跳 $\Delta\mu$ 划分光学厚度/针孔风险：`low`（$< 1.5$）、`moderate`（$[1.5, 2.5)$）、`high`（$\ge 2.5$）。
- **代码实现锚点**：
  - `src/xafs/adapters/athena_selfabs.py :: booth_self_absorption`
  - `src/xafs/core/fit_single.py :: _self_absorption_diag`

---

### `[EQ-06]` 纳米颗粒球截断配位数模型（Calvin / Beale）
- **数学表达式**：
  $$\frac{N(r_1, d_{\text{eff}})}{N_{\text{bulk}}} = 1 - \frac{3}{2}u + \frac{1}{2}u^3, \quad u = \frac{r_1(\text{\AA})}{d_{\text{eff}}(\text{\AA})} \in (0, 1)$$
- **歧义消除约定**：
  - 单位必须统一为埃（$d_{\text{Å}} = 10 \cdot d_{\text{nm}}$）；
  - 半球形基底（`shape="hemispherical"`）等体积等效球径 $d_{\text{eff}} = d_{\text{Å}} / 2^{1/3}$（反演时对应半球直径乘 $2^{1/3}$）；
  - 当 $u \ge 1.0$（粒径不超过第一壳键长）时立即抛出 `XAFSConfigError`。
- **代码实现锚点**：
  - `src/xafs/adapters/artemis_stats.py :: estimate_nanoparticle_size`
  - `src/xafs/core/fit_conditional_cn.py :: parse_defect_model`

---

### `[EQ-07]` 受约束多重散射（MS）与静态劈裂/非高斯热力学修正
- **物理约定**：
  1. **四腿同原子相干往返路径**（$M \to L_1 \to M \to L_1 \to M$）：热位移完全相干，$\sigma^2_{\text{MS}} = (R_{\text{eff}}/r_1)^2 \sigma^2_1 \approx 4\sigma^2_1$；异原子对称链与三腿聚焦路径保持 $\sigma^2_{\text{MS}} = (R_{\text{eff}}/r_1)\sigma^2_1$。
  2. **子壳层静态劈裂方差**（合并容差 $\le 0.20\text{ \AA}$）：
     $$\sigma^2_{\text{geom}} = \frac{\sum_m N_m (R_{\text{eff},m} - \bar{R}_{\text{eff}})^2}{\sum_m N_m}$$
  3. **非高斯累积量一阶热力学键长偏移**（Bunker 有效分布权重近似，$\lambda_{\text{eff}} = 8.0\text{ \AA}$）：
     $$\Delta R_{C_1} = \frac{2\sigma^2}{r_0}\left(1 + \frac{r_0}{\lambda_{\text{eff}}}\right), \quad r_{\text{thermodynamic}} = r_0 + \Delta R_{C_1}$$
     并检验 Gram–Charlier 负值截断（`gram_charlier_clipped`）与单峰性（`unimodal`）。
- **代码实现锚点**：
  - `src/xafs/core/fit_shell_plan.py :: _collinear_ms_rows`、`select_shell_paths`
  - `src/xafs/adapters/artemis_stats.py :: reconstruct_pair_distribution`
