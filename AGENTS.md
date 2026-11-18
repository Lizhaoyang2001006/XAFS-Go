# AGENTS.md — 自动化 Agent 与开发审查硬约束规范

> **适用对象**：所有参与本项目代码编写、重构、审查的 AI Agent（大语言模型）、自动化脚本及核心开发者。  
> **核心宪法**：**物理守恒高于拟合残差；遇到异常如实降级，严禁放宽门禁掩盖失败；严禁破坏向后兼容与基线一致性。**

---

## 一、 五大绝对不可逾越的硬约束红线（Zero-Tolerance Rules）

### 红线 1：12 个锁定元素常量绝对不可动
* 文件位置：`absorber.py` 的静态元素定义表；
* **锁定元素清单**：`Cr, Mn, Zn, Mo, Ru, Rh, Pd, Ag, Ir, Au, Ce, U`；
* **硬约束要求**：上述 12 个元素的静态定义必须严格保持 `s02=None, calibrated=False`。严禁任何 Agent 擅自填入经验默认值或试图放宽验证；Fe 氧化物生产值锁定为 `0.803`，严禁替换；
* **优先级秩序**：$S_0^2$ 解析链必须严格遵循 `CLI显式 (--s02)` > `--chemical-state` > `--s02-from-t5` > `absorber.py 静态表`。

### 红线 2：门禁常量不可放宽，拟合失败必须如实降级
* **严禁放宽常量**：
  * 全谱拟合可信门限：$R\text{-factor} \le 0.08$（红灯硬拦截，绝不可放大至 0.10 以上）；
  * 吸收零点偏移门限：$|\Delta E_0| \le 15.0\text{ eV}$（拦截非物理发散）；
  * 结构识别评分：`CONFIDENT_SCORE = 0.16`、`CONFIDENT_MARGIN = 0.03`；
  * 边后归一化平台：严格处于 $[0.85, 1.15]$ 区间；
  * 荧光自吸收硬红线：浓样 $\Delta\mu \ge 1.0$ 未校正自吸收时，强制触发 `cn_gate=True`，关闭 CN 宣称（`cn_reliable=False`）。
* **处理准则**：当拟合指标超出上述合理范围时，**唯一合规操作是如实标记不可靠并降级（退守 L1 第一壳层）**。严禁 Agent 通过“调宽门限参数、修改常量容差”来换取表面的拟合通过。

### 红线 3：自由度与独立点数物理红线（Stern 规则）
* 拟合前必须严格校验自由变量数与独立数据点数关系：
  $$N_{\text{idp}} = \frac{2\Delta k\Delta R}{\pi}$$
  （**严格遵循 IFEFFIT / larch / Artemis 标准，不加 2**。$+2$ 只是部分教材的
  保守端点修正：在代码层强制加 2 会虚增独立点数、放走过拟合，且会大面积
  破坏标准标样的回归基线。严禁任何 Agent 在 `artemis.independent_points`
  或任何拟合自由度计算中额外 $+2$。）
* 自由度冗余约束：$N_{\text{var}} \le \frac{2}{3} N_{\text{idp}}$ 为**强建议（Warning）**；
  绝对红线保持 $N_{\text{var}} < N_{\text{idp}}$（严禁 $N_{\text{var}} \ge N_{\text{idp}}$）；
* 在 `select_shell_paths` 中超出自由度预算的高阶壳层必须自动截断并记入 `truncated`，严禁 Agent 解除截断。

### 红线 4：多壳层 $\Delta E_0$ 全局单一变量绑定
* 在多壳层拟合时，所有路径必须绑定全局单一 $\Delta E_0$（1 个独立自由变量）；
* 严禁将各壳层的 $\Delta E_0$ 独立放开拟合，防止光电子动量与键长 $R$ 发生虚假数值耦合。

### 红线 5：零退化与 CI 基线逐位一致
* 任何代码改动、重构或优化，必须保证 `run_ci.py` 与 `regression_test.py` 全绿通过；
* 标准标样（如 Lepidocrocite、Fe foil）的基准拟合结果、指标导出数值必须保持历史逐位一致，不可发生数值漂移；
* 所有输出文件落盘逻辑必须保持防静默覆盖机制，保留用户既有分析记录。

---

## 二、 模块架构分工与修改边界

| 模块文件 | 核心职责 | Agent 修改边界与注意事项 |
|:---|:---|:---|
| `run_xafs.py` | CLI 入口、参数解析、管线调度、产物落盘 | 仅允许增加参数别名或辅助开关，不得破坏既有默认参数行为；必须支持向后兼容。 |
| `xafs_pipeline.py` | S1-S6 预处理主干编排 | 严格保证数据处理流的确定性，各阶段数据矩阵必须有严格的非空和有效值校验。 |
| `structure_id.py` | S6.5 结构指纹识别与分类 | 识别阈值 0.16 与 margin 0.03 严禁修改；未解释显著峰强制降级为 partial。 |
| `structure_lib.py` | 结构库检索与动态催化 Motif 生成 | 动态 SAC/DAC/Alloy 缓存应使用哈希隔离，不得污染主库静态索引。 |
| `xafs_fit.py` | S7 Artemis 拟合编排与物理诊断 | 壳层合并容差固定为 0.20 Å；权重拟合严格基于 Softmax；反常参数必须降级。 |
| `artemis.py` | 空间拟合底层数值求解引擎 | 协方差、雅可比矩阵求逆与相关系数计算必须保持数值稳定，不随意更改阻尼因子。 |
| `xafs_report.py` | 报告主调度、HTML 与拟合参数表导出 | 必须严格区分 FT 表观峰位与拟合真实键长 $R$；禁止恢复过程调试图或破坏 7 列 `_fit_table.csv` 基线。 |
| `xafs_plots.py` / `xafs_plot_special.py` | 统一催化剂结构剖析图件、特殊诊断图与 `summary.csv`/Markdown 表 | 通过 `--plot {none,structure,full,paper}` 统一调度；严禁修改或放宽门禁结论展示。 |

---

## 三、 提交审查前自检清单（Pre-flight Checklist）

在完成任何代码修改、生成 PR 或向用户汇报前，Agent 必须在内心依次打勾核验：

- [ ] 是否未修改 `absorber.py` 中的 12 个锁定元素（保持 `s02=None`）？
- [ ] 是否未放宽任何 $R\text{-factor}$、$\Delta E_0$、结构评分或自吸收门禁阈值？
- [ ] $N_{\text{idp}}$ 是否仍按 IFEFFIT 口径 $2\Delta k\Delta R/\pi$ **不加 2**，且未出现 $N_{\text{var}} \ge N_{\text{idp}}$？
- [ ] 拟合失败时是否已如实降级退守 L1，而非掩盖错误？
- [ ] 多壳层拟合是否依然保持全局唯一 $\Delta E_0$ 绑定？
- [ ] 本地运行 `python run_ci.py` 是否 100% 通过且无回归退化？
- [ ] 报表输出是否严格区分了表观 FT 峰位与相位修正键长 $R$？
- [ ] 面向实际样品输出结论时，是否已按 `user_guid.md` 终局八步核验清单核对能量标尺、高 $k$ 噪信比、WT $k$-空间秩序与边界截断（严禁把贴边或重壳层 $k_{\rm peak} \le$ 第一壳轻配体的结果标为 WT 确认）、$k^1\text{–}k^3$ 趋势、外壳层 F-test $\ge 95\%$ 与门禁卡一致性？

---

## 四、 物理白盒锚定与仓库分层治理准则

### 1. 公式编号强制双向锚定
* 项目核心物理公式、符号方向与单位制唯一权威定义于 [PHYSICS_SPEC.md](./PHYSICS_SPEC.md)（`[EQ-01]` 至 `[EQ-07]`）；
* 涉及核心物理公式实现的函数必须在 Docstring 中标注 `[Implements: EQ-XX]` 双向锚点，变量命名、单位制（eV / Å⁻¹ / Å / Å² / rad）与常数必须与之一一对应；
* 修改拟合、傅里叶变换或自由度逻辑后，不得仅凭代码自洽或拟合优度宣称“物理正确”，必须通过解析解探针测试（[tests/test_analytical_probes.py](./tests/test_analytical_probes.py)）检验峰值位置、配位振幅与 $N_{\text{idp}}$ 守恒。

### 2. 单向依赖与防污染目录规范
* **物理目录划分与单向调用层级**：
  1. `src/xafs/domain/`：`xafs_core.py`, `absorber.py`, `xafs_config.py`, `structure_model.py`（契约与常量底座）
  2. `src/xafs/adapters/`：`athena*.py`, `artemis*.py`, `larch_wt.py`（`xraylarch` / `xraydb` / FEFF 隔离层）
  3. `src/xafs/io/`：`xafs_io.py`, `xafs_writer.py`, `cif_gate.py`, `phase_*.py`, `*_provenance.py`（输入输出与溯源层）
  4. `src/xafs/core/`：`xafs_pipeline.py`, `structure_*.py`, `catalytic_builder.py`, `fit_*.py`, `xafs_fit.py`, `xafs_lcf.py`, `calibrate_s02.py`, `cn_*.py`（核心流程与物理诊断）
  5. `src/xafs/cli/`：`xafs_report.py`, `xafs_plots.py`, `xafs_plot_special.py`, `paper_figures.py`（统一结构剖析出图与报告入口，由根目录 `run_xafs.py` 调度）
  * **严禁底层反向依赖上层编排、CLI 或 `scripts/`/`tests/`**；所有内部路径解析统一通过 `src/xafs/paths.py` (`REPO_ROOT`)，禁止在子包中硬编码相对母目录。
* **禁止“文件末尾打补丁”与僵尸代码**：修改物理逻辑必须原地替换，严禁在文件末尾追加 `*_v2` 或保留同名死代码覆盖旧实现；
* **实验草稿与缓存隔离**：独立批处理/标定/压测脚本放 `scripts/`，临时试错脚本统一放在 `sandbox/`，FEFF 与动态构型缓存隔离在 `_feff_cache/` / `_structure_lib/_dynamic_cache/`，严禁提交进 Git。

---

## 五、 Agent 交互与执行铁律（Silent Execution & Minimal Diff）

1. **沉默执行协议**：除非缺少运行所需的外部实验文件或环境变量，严禁向用户提出无关的开放性反问（如“是否顺便重构某模块”）。局部实现细节歧义直接依据本规范与 `PHYSICS_SPEC.md` 做出唯一确定性工程决策。
2. **最小差异修改（Minimal Diff）**：每次只修改工单目标函数或文件；严禁未经要求顺手重构无关模块的风格、注释或导入顺序。
3. **先跑后说（Execution-First）**：任何代码修改后必须先在终端实际运行对应验证命令，汇报时只陈述已真实观测到的命令输出与退出码，严禁未运行测试即宣称通过或靠记忆补全数值。
4. **两轮熔断原则**：同一问题连续修改 2 轮仍未通过检验，立即停止堆叠补丁，如实汇报阻塞堆栈与失败边界。

---

## 六、 L0–L3 四级分级验收体系

| 级别 | 定位 | 验收命令 / 范围 | 执行时机 |
|:---|:---|:---|:---|
| **L0** | 静态编译与入口/物理不变量快检 | `python -m py_compile` 覆盖 `src/`/`scripts/`/`tests/` 及 L0/L1 纯数值探针 | 每次代码修改后立即执行 |
| **L1** | 专项物理单测、解析解探针、绘图与审计 | `python -m unittest tests/test_physics_p01_p07.py tests/test_analytical_probes.py tests/test_plots_structure.py tests/test_data_audit.py` | 每次 `commit` 前必须全绿 |
| **L2** | 全量契约回归 | `python regression_test.py` | 模块改动合并主干前 |
| **L3** | 全流程 CI 与真实标样留出验证 | `python run_ci.py`、`python scripts/validate_real_cn.py --outdir _real_cn_validation` | 版本封版 / 打 Tag 前 |
