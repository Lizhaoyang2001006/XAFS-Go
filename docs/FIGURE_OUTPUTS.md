# XAFS 结构解析导向输出图与报表规范

所有物理结论与门禁常量（`R-factor <= 0.08`、`|ΔE0| <= 15 eV`、Stern `N_idp = 2ΔkΔR/π`、全局唯一 `ΔE0` 等）严格以 `AGENTS.md` 与 `PHYSICS_SPEC.md` 为唯一准则。本系统输出**直接服务于催化剂局域结构剖析**的图件与表格架构。

---

## 1. 入口与统一切换开关

两个 CLI 入口（`run_xafs.py` 与 `src/xafs/cli/xafs_report.py`）与内部独立模块 `src/xafs/cli/paper_figures.py` 协同工作：

| 入口 | 默认 `--plot` | 主要职责 |
|:---|:---|:---|
| `run_xafs.py` | `none` | 批处理/多元素/定量拟合主引擎；显式加 `--plot structure` / `full` / `paper` 即可出图 |
| `python -m xafs.cli.xafs_report` / `xafs_report.py` | `structure` | 完整结构剖析与报告入口：自动 S1–S7 + 结构总图/壳层鉴别图 + 拟合表 + HTML |
| `paper_figures.py` | — | 对已有 `fit.json` 重绘论文发表级组合图（已升级 R 空间逐路径分壳拆解曲线） |

统一绘图与报表 CLI 参数（`run_xafs.py` 与 `xafs_report.py` 完全对齐）：

| 参数 | 可选值 / 默认 | 说明 |
|:---|:---|:---|
| `--plot` | `none` / `structure` / `full` / `paper` | `run_xafs.py` 默认 `none`；`xafs_report.py` 默认 `structure` |
| `--plot-formats` | 默认 `png`（支持 `png,pdf`） | 同时导出高分辨率栅格与矢量格式 |
| `--plot-dpi` | 默认 `300` | 输出图像分辨率 |
| `--phase-correct` | 布尔开关 | 额外单独导出 `<stem>_phase_corrected.png/.csv`（注：`structure` 档位已内置相位修正对比面板） |
| `--chi2-contour` | 布尔开关 | 按需导出第一壳 $(N,\sigma^2)$ 二维 $\Delta\chi^2$ 置信椭圆（`1σ/2σ/3σ`） |
| `--operando` | 布尔开关 | 至少 3 条时间序列谱：输出 `<stem>_operando_evolution.png` 与组分/浓度表 |
| `--no-html` | 布尔开关 | 仅 `xafs_report.py`：关闭单文件 `<stem>_report.html` 生成 |

---

## 2. 四类催化剂结构剖析核心图件

### 图 1：`<stem>_structure_main.<ext>` — 催化剂配位结构剖析总图（6 面板 `2×3`）

核心用途：一张图回答“**吸收中心是什么价态、数据信噪与 k 空间特征如何、实空间每一层归属于谁、物理门禁是否放行**”。

- **(a) Normalized XANES & White Line**：以 $E - E_0$ 为横轴展示吸收边与白线位置（价态、d 带空穴）。
- **(b) First Derivative $d\mu_{\rm norm}/dE$**：突出边前 1s→3d 特征峰与吸收边肩峰分裂。
- **(c) $k^w\chi(k)$ 拟合与残差**：叠加拟合区间底色与下置残差线，检验高 $k$ 段重原子振荡覆盖情况。
- **(d) R-space Fit & Path-by-Path Shell Decomposition**：**核心结构剖析面板**——在实验 $|\chi(R)|$、总拟合 $|\chi(R)|$、实部 $\mathrm{Re}\,\chi(R)$ 与模残差之外，逐条叠加拟合路径的独立实空间贡献 $|\chi_i(R)|$（如 `M-N`、`M-C`、`M-M`）。
- **(e) Morlet Wavelet $|W(R,k)|$**：二维 $(k,R)$ 空间直观区分第一壳轻配体（低 $k$）与第二壳重散射原子（高 $k$）。采用各拟合路径 FEFF 理论峰紧窗锚定，并在**轻/重双向防错标**下运行：若轻散射外壳层被同模型邻近重散射壳层的高 $k$ 峰脊掩蔽，则取消误导性峰圈并标示 `[masked by ...; th k≈...]`；若轻壳层单独落入高 $k$ 重原子区而无重壳层解释，则以红色告警标为 `[mismatch!]` 并阻断门禁 `PASS`。
- **(f) Fitted Structure & Physical Gates Card**：汇总样品信息、Top-1 结构归属、各路径 $(N, R, \sigma^2)$、$R\text{-factor}$、$\Delta E_0$、$N_{\rm var}/N_{\rm idp}$、$S_0^2$ 来源与红绿灯门禁告警。**严禁假 PASS**：仅当 `cn_reliable=True`、`cn_claim_allowed` 不为 `False`、`diagnostic_only=False`、`cn_status` 非假设/诊断态，且 R-factor/$\Delta E_0$/相关性 $|\rho|<0.90$/$\sigma^2$/Hamilton F-test（$\ge 95\%$）/WT $k$-空间健康检查（`wt_k_sanity`）全部无告警时，才允许显示 `PASS — quantitative CN allowed`；否则如实标记为 `DIAGNOSTIC/HYPOTHESIS ONLY` 或 `FAIL / DOWNGRADE` 并逐项列出告警。

### 图 2：`<stem>_shell_dissection.<ext>` — 壳层轻重鉴别与真实键长核验图（3 面板 `1×3`）

核心用途：**破解单一表观 FT 图对第二壳层轻/重原子及真实键长的误读**。

- **(a) Multi-$k$-weight FT ($k^1 / k^2 / k^3$ normalized)**：利用轻原子（C/N/O）背散射振幅偏向低 $k$、重金属/载体阳离子（Fe/Co/Ni/Pt/Ce）偏向高 $k$ 的物理性质，若第二壳在 $k^3$ 权重下相对增强，则指向金属–金属/重载体配位；若反而持续萎缩，则会触发与重散射假设相矛盾的告警。
- **(b) True Bond Length Check（Apparent $R$ vs Phase-Corrected $|\chi(R)|$）**：应用 FEFF 散射相移校正，直接把偏短 0.3–0.5 Å 的 FT 表观峰拉回真实原子间距 $R$，并标出拟合键长参考线。
- **(c) Shell 1D WT $k$-profiles vs Reference Element Rulers**：优先依据各拟合路径的独立 FEFF 表观峰位 $R_{\rm feff}$ 在紧窄窗内锚定表观 $R_{\rm app}$（避免相邻峰串扰），抽取 1D WT $k$-轮廓并叠加理论参考元素竖线标尺。当检测到某壳层 $k_{\rm peak}$ 贴近变换边界时标为 `[edge!]`；当重原子外层（$Z \ge 20$）出现 $k_{\rm peak} \le$ 第一壳轻配体 $k_{\rm peak} + 0.3$、边界截断或 $\sigma^2 > 0.010\text{ Å}^2$ 时，在面板顶部自动弹出醒目告警徽章。

### 图 3：`<stem>_candidates_compare.<ext>`（`--plot full` 且存在结构识别排名时输出）

- **(a) R-space & (b) k-space Candidate Comparison**：将实验谱与 S6.5 排名 Top-K 竞争构型（例如 SAC vs 纳米团簇 vs 体相氧化物）理论归一化谱叠加，直观评估“为何判定为候选 #1 而排除竞争构型”。

### 图 4：按需专场景扩展图

- **`<stem>_rdf_distribution.<ext>`**（当 `--vary-c3` 或 `--vary-c4` 启用且参与拟合时自动输出）：重构非高斯配对径向分布函数 $g(R)$，标注众数键长 $R_{\rm mode}$、均值键长 $R_{\rm mean}$、Bunker–Tranquada 热力学平均键长 $\langle R_{\rm th}\rangle$ 及 `clipped` / `non-unimodal side-lobe!` 非物理告警。
- **`<stem>_chi2_contour.<ext>`**（`--chi2-contour`）：第一壳 $(N, \sigma^2)$ 二维 $\Delta\chi^2 = 2.30 / 6.18 / 11.83$（1σ/2σ/3σ）置信椭圆，直观评估参数简并程度。
- **`<stem>_operando_evolution.<ext>`**（`--operando`）：原位序列四宫格演化图（MCR 纯组分分数 / LCF 标样权重 / $E_0$–白线描述符 / 纯组分谱），以及可选 EXAFS 轨迹演化图 `write_operando_exafs_diagnostics`。
- **`<stem>_paper.<ext>`**（`--plot paper` 或 `paper_figures.py`）：期刊投稿紧凑多面板组图；R 空间拟合面板 `(d)` 现已原生叠加逐路径独立 $|\chi_i(R)|$ 分壳曲线。

---

## 3. 结构定量表与机器可读台账

| 文件 | 格式 | 核心用途 |
|:---|:---|:---|
| `<stem>_fit_table.csv` | CSV (UTF-8 BOM) | 定量拟合参数表（**严格维持 7 列历史冻结基线格式不变**） |
| `<stem>_fit_table.md` | Markdown | 同口径拟合参数表 Markdown 版，可直接粘贴至论文/实验记录 |
| `<stem>_fit_table.png` | PNG | `xafs_report.py` 输出的 Table S 风格拟合表图（若仅拟合 C3 时自动扩展示出 C3 列） |
| `<stem>_summary.csv` | CSV (UTF-8 BOM) | 催化剂结构归属（`structure_top1`、`structure_status`）、门禁状态、$S_0^2$ 来源、$N_{\rm var}/N_{\rm idp}$、WT 壳层 $k$-空间健康核验（`wt_k_sanity`）、预处理自审计（`data_quality_audit`：`.prj eshift`/单扫高 $k$ 噪声/`Rbkg` 入侵）、外壳层 F-test 显著性与静态劈裂等紧凑汇总表 |
| `<stem>_indicators.csv` | CSV (UTF-8 BOM) | 边前峰/白线/E0/Rbkg/F-test/相关性等全量物理指标 |
| `<stem>_meta.json` | JSON | 完整运行台账（`params` / `gates` / `decisions` / `formal_cn` / `run_manifest`） |
| `<stem>_structure_id.json` | JSON | 全量候选结构评分排名与覆盖检查 |
| `<stem>_report.html` | HTML（单文件内联） | 汇总卡片 + 门禁结论 + 拟合表 + 结构主图/壳层鉴别图内联报告 |
| `<stem>_{mu,xanes,chik,chir,chiq,wavelet}.csv` | CSV | 与图件面板一一对应的原始处理光谱数据 |

> 所有正式产物通过 `src/xafs/io/xafs_writer.py` 原子写盘，同一目录重复运行时旧文件自动备份至 `.backup/<时间戳>/`，严禁静默覆盖。单测校验绘图与审计合约可运行 `python tests/test_plots_structure.py` 与 `python tests/test_data_audit.py`。
