# XAFS 光谱处理与定量分析管线 (XAFS Processing & Fitting Pipeline)

> **核心承诺**：基于严谨物理门禁的 XAFS/EXAFS 自动化分析管线。**不放宽门禁、不产生虚假拟合优度、遇到异常如实降级**，全面支持常规体相材料及前沿催化非理想结构（单原子 SACs、界面负载、纳米合金、掺杂固溶体），确保学术成果具备最高等级的可复现性。

---

## 🚀 极速体验（1 分钟上手）

```bash
# 安装依赖
pip install -r requirements.txt

# 一行命令跑完完整管线（数据预处理 + 结构识别 + 定量 CN 拟合 + 催化剂结构图件）
# 示例使用仓库自带的 Fe lepidocrocite 标样（无需另外下载数据）
python run_xafs.py "data/练手数据/02_Fe_standards/Fe_lepidocrocite.000" --absorber Fe --fit --plot structure -o xafs_out/sample_Fe
```

所有产物统一落在 `xafs_out/sample_Fe/`。重复运行时既有产物会先备份到
同目录 `.backup/<时间戳>/` 再写入（防静默覆盖），核心重点查看以下文件（详见 [`docs/FIGURE_OUTPUTS.md`](./docs/FIGURE_OUTPUTS.md)）：
* `*_structure_main.png`：催化剂结构剖析 6 面板主图（XANES、$d\mu/dE$、$k^w\chi(k)$ 拟合与残差、$R$ 空间拟合+逐路径分壳拆解、Morlet WT、门禁与结构汇总卡）；
* `*_shell_dissection.png`：壳层轻重鉴别与真实键长核验图（$k^1/k^2/k^3$ 多权重对比、相位修正前后 $|\chi(R)|$、逐壳 1D WT 轮廓）；
* `*_fit_table.csv` / `.md` / `.png` & `*_summary.csv`：定量拟合参数表（FEFF 相位修正真实键长 $R$、配位数 $N$、$\sigma^2$）与门禁摘要表；
* `*_indicators.csv`：关键物理门禁核验表（$R\text{-factor}$、$\Delta E_0$、$S_0^2$ 来源、自吸收状态）。

---

## 📚 文档全景导航（按角色查阅）

为避免信息冗余与口径脱节，本仓库建立了分工明确的文档体系：

| 读者角色 | 推荐必读文档 | 包含核心内容 |
|:---|:---|:---|
| 🔬 **实验与科研人员** | [**`user_guid.md`**](./user_guid.md) | **用户操作总指南**：3 分钟跑通、结果红黄绿灯三步判读法、避坑决策树、标准样标定及论文表格规范填报。 |
| 🧪 **复杂催化体系进阶** | [**`docs/CATALYTIC_MOTIF_PIPELINE.md`**](./docs/CATALYTIC_MOTIF_PIPELINE.md) | **非理想结构建模执行方案**：单原子（SACs）、界面负载（如 $Pt/CeO_2$）、晶格掺杂及非高斯静态无序的局域点阵建模与物理约束。 |
| 📐 **物理公式白盒规范** | [**`PHYSICS_SPEC.md`**](./PHYSICS_SPEC.md) | **数理与单位制唯一权威字典**：`[EQ-01]`–`[EQ-07]` 公式约定、Stern $N_{\text{idp}}$ 定义、正负号与双向代码锚点。 |
| 📊 **鉴别方法与实战经验** | [**`docs/XAFS_SPECIATION_METHODOLOGY.md`**](./docs/XAFS_SPECIATION_METHODOLOGY.md) | **化合态与局域结构六步鉴别逻辑**：XANES/LCF + 多 $k$ 权重 + WT + 嵌套 Hamilton F-test 判据（附 Agent 注意事项 [docs/AGENT_OPERATIONS_NOTES.md](./docs/AGENT_OPERATIONS_NOTES.md)）。 |
| 🤖 **AI 模型与核心开发者** | [**`AGENTS.md`**](./AGENTS.md) | **代码架构宪法**：12 锁定元素不可动、门禁常量不可放宽、Stern 独立点数硬截断、L0–L3 分级验收与 Agent 执行铁律。 |

---

## 🌟 核心特性与设计原则

1. **绝对守恒的物理门禁（Anti-Overfitting）**：
   * 独立点数预算门禁：严格执行 Stern 规则 $N_{\text{idp}} = \frac{2\Delta k\Delta R}{\pi}$（IFEFFIT / larch / Artemis 标准口径，**不加 2**；$+2$ 只是部分教材的保守端点修正，在代码层强制加 2 会虚增独立点数、放走过拟合并破坏标准标样回归基线）；$N_{\text{var}} \le \frac{2}{3}N_{\text{idp}}$ 为强建议（Warning），绝对红线为 $N_{\text{var}} < N_{\text{idp}}$（严禁 $N_{\text{var}} \ge N_{\text{idp}}$），超出自由度预算的壳层强制截断，绝不欠约束过拟合；
   * 全局 $\Delta E_0$ 绑定：多壳层拟合严格共用单一 $\Delta E_0$ 自由度，杜绝壳层独立浮动导致的假键长；
   * 荧光自吸收硬红线：浓样 $\Delta\mu \ge 1.0$ 未校正自吸收时，强制标记 `cn_reliable=False` 并拒绝定量宣称。
2. **催化前沿非周期性体系原生支持**：
   * 内建动态拓扑生成器，支持 `--catalytic-model "sac:N4@2.00,axial=O@2.10,C@2.90"` 直接构建单原子局域点阵；
   * 原生支持金属纳米颗粒粒径从第一壳配位数降幅自动反推（`nanoparticle_size`）；
   * 混合相多位点采用 Softmax 权重参数化拟合（`--fit-weights`），保证相分数恒正且和为 1。
3. **真实键长与表观峰位防伪分离**：
   * 严格执行 FEFF 相位修正，论文表格严格采用修正后的真实 $R$，明确隔离未校正相移的 FT 表观距离（系统偏短 0.3~0.5 Å）。

---

## 🛠️ 项目结构与五层分层目录（已按物理职责归类）

```text
xafs-auto-pipeline/
├── AGENTS.md                         # 面向开发与 AI 模型的硬约束宪法
├── PHYSICS_SPEC.md                   # 核心物理公式与单位制白盒规范 ([EQ-01]–[EQ-07])
├── README.md                         # 项目总览与快速上手
├── user_guid.md                      # 面向科研人员的完整操作指南
├── pyproject.toml / requirements.txt # 工程配置与科学计算依赖锁定
├── run_xafs.py                       # 主 CLI 入口（数据预处理 + 识别 + 定量 CN 拟合）
├── run_ci.py                         # CI 入口（静态编译 + 回归 + 压力测试）
├── regression_test.py                # 全量契约回归测试入口
│
├── src/xafs/                         # 【核心源码库】（严格单向向下依赖）
│   ├── paths.py                      # 仓库根目录权威路径定位
│   ├── domain/                       # [第1层] absorber.py, xafs_core.py, xafs_config.py, structure_model.py
│   ├── adapters/                     # [第2层] athena*.py, artemis*.py, larch_wt.py (xraylarch/FEFF 防腐层)
│   ├── io/                           # [第3层] xafs_io.py, xafs_writer.py, cif_gate.py, phase_*.py, *_provenance.py
│   ├── core/                         # [第4层] xafs_pipeline.py, data_audit.py, structure_*.py, catalytic_builder.py, xafs_fit.py, fit_*.py, calibrate_s02.py, xafs_lcf.py, cn_*.py
│   └── cli/                          # [第5层] xafs_report.py, xafs_plots.py, xafs_plot_special.py, paper_figures.py
│
├── scripts/                          # 【独立运行脚本区】calibrate_t5.py, cif_crawler.py, env_check.py, validate_real_cn.py
├── tests/                            # 【质检测试区】契约回归、物理修复、解析解探针、绘图与预处理审计单测
├── tools/                            # 【构建期工具】phase pack 与标准谱来源池构建
├── docs/                             # 【专题文档】结构剖析图表规范、催化建模、报告标准与方法论
└── data/                             # 【标样与基准数据】
```

---

## 🧪 自动化测试与质量保障

在修改任何代码后，按 [AGENTS.md](./AGENTS.md) 的 L0–L3 分级验收体系执行检验确保零退化：

```bash
# L1：物理单元测试、解析解探针、绘图与预处理审计单测
python -m unittest tests/test_physics_p01_p07.py tests/test_analytical_probes.py tests/test_plots_structure.py tests/test_data_audit.py

# L2/L3：完整 CI（静态编译 + 回归 + 单测）与真实标样留出验证
python run_ci.py
python scripts/validate_real_cn.py --outdir _real_cn_validation
```

---

## ⚠️ 已知限制（Known Limitations）

以下为**刻意保留的现状**，不是待修缺陷。算法层不得凭空捏造物理常数，
也不得为凑门禁而写死默认值；读者在引用定量结论前须自行核对。

| 项 | 现状 | 对结论的影响 / 使用要求 |
|:---|:---|:---|
| 标准谱未随仓库提交 | `_standards_lib/`、`data/standards_raw/` 因许可未明确而未提交。`data/s02_calibration/REPORT.md` 中 12/12 标定的 Rh / Pd / Ag / Ir / Ce / U 六条只保留数值与源文件 SHA256，干净克隆上**无法重算** | 这 6 条应视为 `calibrated_unrepeatable`；需要复核时须自行取得 IIT-Lytle 原始谱并按 `calibrate_t5.py` 协议重跑 |
| Fe 的 $S_0^2=0.803$ | `absorber.py` 注释明确记为"氧化物价态验收锚定的跨材料折中值（原始流程不可复原）"；金属 Fe 另有箔交叉标定值 0.6566 | 该值锁定不改，但**金属相必须**显式 `--chemical-state metal`，否则 Fe 的 CN 系统性偏高约 22%（CN 与 $S_0^2$ 线性耦合） |
| 12 个未标定元素 | Cr / Mn / Zn / Mo / Ru / Rh / Pd / Ag / Ir / Au / Ce / U 静态表恒为 `s02=None, calibrated=False` | 自动 CN 定量会被**拒绝启动**；须显式 `--s02` / `--chemical-state` / `--s02-from-t5`，不得填入经验默认值 |
| T5 标定的平台越界例外 | Pd 标样平台 0.8138（低于 $[0.85,1.15]$ 下界）、CeO₂ 荧光计数谱平台 ≈1.20 / 2.271，属于基体吸收与荧光计数归一化带来的客观物理现象；标定链用 `strict=False` 手工豁免 | **生产链默认 `strict=True` 不受影响**；`--no-strict` / `strict=False` 是明确的权责划分——使用者一旦豁免，须自行对归一化与后续 CN 负责，并在论文中披露 |
| L 边 $k_{\max}$ 物理截断 | Ce L3(5723 eV) → L2(6164 eV) 仅 441 eV，$k_{\max}=\sqrt{0.2624685\times441}=10.76\ \text{Å}^{-1}$；静态 ceiling 取 10.2 | 主管线在 `process_sample` 入口按元素裁剪 `--kmax`，超限时打印并把裁剪记录写入 meta；显式传大 kmax 也不会越过物理上限 |
| 冷启动依赖 | 仓库不含 FEFF 可执行二进制（依赖 `xraylarch` 的 Fortran `feff6l()`）；`_structure_lib/_phase_cache/`、`_dynamic_cache/`、`data/raw/` 等缓存目录由 `.gitignore` 排除 | 新 motif 需现场跑 FEFF；干净克隆需先 `pip install -r requirements.txt` 并按需重建缓存，部分结构库检索需联网 |
| 催化团簇第一壳"配位残缺" | 早期 `catalytic_builder.py` 的局部团簇构建可能因 `rcut` 截断过早 / 超胞扩展不足产生**边缘效应**，看起来像配位数偏少 | **不是空间群未展开**：`artemis.read_cif_sites` 已用 gemmi 的 `make_small_structure_from_block` + `get_all_unit_cell_sites()` 完整展开对称等价位。定量结构请优先复用该路径，团簇生成时给足 `rcut` 与超胞 |

> 主管线产物（`run_ci.py` 的 `ci_report.*`、两个 CLI 的 JSON/CSV/HTML/图包/窗口
> 调节器、诊断 CSV、正式 `cn_table.csv`）统一走 `xafs_writer.py`：重复运行同一
> `--outdir` 时，既有文件会先备份到 `.backup/<时间戳>/` 再写入，不静默覆盖
> （临时目录跳过备份）；原子写 + 写锁保证多线程并发不会留下半截文件。
> 离线标定/爬取工具（`calibrate_*.py`、`cif_crawler.py`、`tools/`）与 FEFF
> 临时工作目录仍直接写盘：它们不产出正式报告，也不共用 `--outdir`。
