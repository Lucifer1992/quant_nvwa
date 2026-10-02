# 大模型看 K 线策略（VLM Strategy）可行性评估

> 配套文档：`docs/quant_platform_design.md` 第 6.4 节。本文对该节做深入论证与结论修订。
> 结论口径：区分 **[论文+代码]** / **[论文摘要级]** / **[社区经验]** / **[宣传，不可验证]**。

---

## 0. 结论先行

**可行，但必须改变它在系统里的角色。**

| 定位 | 可行性 | 依据 |
|------|--------|------|
| ❌ 端到端：K线图图像 → 模型直接给 entry/stop/TP → 下单 | **目前不成立** | 2026 年多模型评测显示 VLM 读图存在**时间视野绑定失败**与方向偏置，作者结论是"不可作独立决策引擎" [论文摘要级] |
| ⚠️ 端到端但降级为"高置信稀疏信号 + 严格 abstain" | 边缘可行，需 Gate 验证后才知 | 选择性预测/弃权机制（FinAbstain 类）能把精度换覆盖率 [论文] |
| ✅ **VLM 作形态与状态识别器（结构标签生成器）** | **推荐路线** | 图像输入在"趋势相关性"上中位数高于纯数值基线，即在**方向性行情中有增益** [论文摘要级]；CNN/ViT 做视觉市场状态分类可达 AUC-ROC 0.892 [论文，无数值基线对比] |
| ✅ **VLM 作战术否决器 / 风控第二意见** | **推荐路线** | 不进收益归因，只降低尾部风险，可证伪成本低 |
| ✅ **LLM 生成因子表达式 + 确定性回测引擎**（AlphaAgent 路线） | 可审计性最好 | 输出是可执行代码，可严格 point-in-time 验证 [论文+代码] |

因此建议把 `vlm_v1` 从设计文档里的"决策器"改为：
**VLM 只输出离散、无量纲的结构化标签（形态、市场状态、有效性判断、可否定），价位与数量一律由数值层（指标/ATR/风控）给出。**

> 关于你提到的"**laya**"：我以 `LADER` / `TLV-2` 等拼写做多轮检索，**未找到可验证的论文或官方代码**（只找到 TradingAgents、FLAG-Trader、MARSHAL 等）。请确认你所指的具体名称或出处，我再补入分析。当前本文以能验证的工作为准。

---

## 1. 证据地图（2024-12 → 2026-10）

### 1.1 正面/中性证据

| 工作 | 输入模态 | 输出 | 报告结果 | 强度 / 已知缺陷 |
|------|---------|------|---------|------|
| FinAgent (KDD24) | 数值 + 新闻 + **K线图图像**，工具化指标 + 双层反思记忆 | 带 CoT 的动作 | Tesla 累计约 92%，超 buy&hold ~36% | [论文+代码]；**工具阈值按股票标定，迁移到加密跌 20%+** → 对本项目是直接警示：ETH 不能复用其指标阈值 |
| MM-DREX | K线图 + OHLCV + 文本，VLM 路由 4 专家 | 专家权重（非订单） | 美股 47.5% / Sharpe 1.83，含 BTC/ETH | [论文]；作者自述**未建模手续费/滑点/延迟** |
| MarketSenseAI 2.0 | RAG + agent，纯数值/文本 | 决策支持 | S&P100 两年 125.9% vs 73.5% | [论文]；成本敏感性未讨论 |
| Trading-R1 / FLAG-Trader | 结构化市场状态 | 交易动作 | Qwen3-**4B** + GRPO 报告高 Sharpe | [论文]；少标的、短窗口，外推弱 |
| Alpha-GPT 2.0 / AlphaAgent (KDD25) | LLM 生成因子表达式 | 可执行 alpha 代码 | 可验证性最好；含 self-provenance 与 **lookahead detection** | [论文+代码]；不是端到端择时 |
| Visual Chart Representations for Crypto Regime | 蜡烛图 / GAF，128×128 | **市场状态分类** | 4 层轻 CNN 最优，AUC-ROC 0.892（BTC/ETH/S&P，2018-2024） | [论文，课程作业级]；**简单表示优于复杂表示**；无与数值基线对比 |

### 1.2 反面证据（这是决策的关键）

| 发现 | 含义 | 强度 |
|------|------|------|
| Do VLMs Truly "Read" Candlesticks? (2026-04) | 多尺度蜡烛图 vs **XGBoost 特征基线**：VLM 仅在**持续单边行情**中泛化，在常态震荡中失效；对 prompt 里显式指定的预测视野**敏感度低** → 时间推理缺陷 + 稳定输出偏置 | [论文摘要级] |
| FinChart-Bench (ACL2026) | 25 个 LVLM 在真实金融图表上指令跟随与空间推理显著不足 | [论文] |
| Are LMs Actually Useful for TS Forecasting? (NeurIPS24) | 去掉 LLM 组件，精度不明显下降 | [论文+代码] |
| FINSABER | LLM 策略优势随时间长度与标的数量扩大而衰减 | [论文] |
| StockBench | 金融问答能力与实盘结果**不相关**；参数规模非决定因素 | [论文+基准] |
| The Alpha Illusion（审计类） | 大量"系统×摩擦"组合未建模；**训练截止后性能衰减 ~71.85%**；短窗口置信区间与收益同量级 | [论文，需自行复核] |
| Profit Mirage / FactFin | 信息泄漏四维量化："回测利润在知识窗口结束后蒸发"；用反事实扰动迫使学因果而非记忆 | [论文] |
| Agentic Trading 综述 | 19 篇主集**无一达到完全可复现级别**，仅 2/19 提供时间一致切分 | [论文] |
| Backtrader-Bench | 工具增强的前沿模型 ~90%，而**约一半被测模型 ≈ 随机(25%)** | [论文] |

### 1.3 从证据推出的三条判断

1. **"看图识别形态"有正证据，"图 → 指定周期择时下单"目前是负证据。** 瓶颈是**数值精度、空间推理、horizon 绑定**，而不是模型不够大。
2. **凡是报告漂亮收益的工作，几乎都没建模交易成本**，而永续合约还有资金费率。你的 MVP 必须自带成本闸门，否则无法和这些工作对话。
3. **LLM 相对简单数值基线没有稳定优势**（NeurIPS24 / FINSABER / StockBench）。所以 `vlm_v1` 的评估标准不能是"它自己赚不赚钱"，而是 **"在 rule_v1 / dl_v1 之上是否带来统计显著的增量"**——这是唯一可证伪的实验设计。

---

## 2. 推荐架构：VLM 降级为标签器/否决器

```
                     ┌────────────────────────────┐
 bars(15m/1h/4h) ──▶ │ 数值特征层 + rule_v1/dl_v1 │──▶ 候选 Signal（含 SL/TP/qty）
                     └──────────────┬─────────────┘
                                    │
        渲染 PNG(近120根, 含成交量/MACD/RSI 副图, 标注候选入场点与结构位)
                                    ▼
                     ┌────────────────────────────┐
                     │ VLM 评估器 (vlm_v1)         │
                     │ 只输出离散标签，禁止给价位: │
                     │  pattern: enum[...]        │
                     │  regime: trending|range|shock
                     │  setup_validity: 0|1       │
                     │  veto: bool + reason       │
                     │  confidence_raw            │
                     └──────────────┬─────────────┘
                                    ▼
                     校准器(温度缩放/isotonic, 用历史样本外拟合)
                                    ▼
                     RiskGate: 标签→调整(降杠杆/降仓位/否决/放行)
                                    ▼
                     PaperPortfolio + 飞书卡片
```

为什么这样切：

- 把 VLM 的**能力区**（视觉形态、市场状态的整体把握）和**能力盲区**（数值精度、价位、时间视野）分离。价位由 ATR/结构位算出，这是确定性代码。
- 输出**离散枚举** → 可以做混淆矩阵、可以做标签一致率测试、可以做反事实扰动测试；输出连续价位则无法归因。
- VLM 不做归因主体，只在**风控通道**上起作用，即使它无效也只是"多一次无操作调用"，不会污染策略逻辑。
- 成本可控：只在有候选信号时才调用 VLM（**信号驱动调用**，而非每根 bar 都调用），调用量下降 1~2 个数量级。

三种候选方案对比：

| 方案 | 延迟 | 单日期成本（ETH, 15m） | 可回测性 | 证据支持 | 结论 |
|------|------|------|------|------|------|
| A 端到端图→下单（每根 bar 调用） | 高 | 高（~96 次/日） | 差（horizon 无法保证） | 负面 | 仅作研究对照组，不进生产 |
| **B 标签器/否决器（信号驱动调用）** | 中 | 低（~5-15 次/日） | **好** | 正面 + 可控 | **MVP 采用** |
| C LLM 生成因子 + 确定性回测 | 低（离线） | 离线一次性 | **最好** | 正面 | 与 `dl_v1` 并行，作为 M5 扩展 |

---

## 3. 基座模型：第三方 API vs 本地 7~8B

### 3.1 可用模型清单（截至 2026-10）

开源可本地部署（图表/时序视觉向）：

| 模型 | 规模 | 许可证 | 特点 | 备注 |
|------|------|--------|------|------|
| **Qwen3-VL 4B / 8B** | 4B/8B | Apache-2.0 | 原生高分辨率 + **坐标 grounding**，图表能力强 | 8B FP16 权重约 17GB，单卡 24GB 可跑但 KV 吃紧，建议 AWQ/限上下文 |
| Qwen2.5-VL-7B | 7B | Apache-2.0 | 生态最成熟，vLLM/SGLang/Ollama 均支持 | 稳妥降级选择 |
| InternVL3.5-8B | 8B | MIT | 文档/图表基准小尺寸档领先（论文口径） | 需以 HF 卡分数复核 |
| GLM-4.5V | 3B / 9B | 见 HF | OCR / GUI grounding 强 | 适合读图上的文字与坐标 |
| Molmo 7B | 7B | 见 HF | **点选式 grounding** 独特，标注数据开放 | RAG 编码对推理框架兼容较晚 |
| 备选 | — | — | Pixtral-12B(无 grounding)、DeepSeek-VL2-Small 3.3B、Phi-4-mm 5.6B | 12B 超单卡舒适区 |

闭源 API 现状：三家（Claude / GPT / Gemini）均支持工具调用与严格 JSON schema 输出，均支持图像输入；Claude 与 Gemini 支持 prompt caching（读缓存约 1/10 价），OpenAI 有 detail=low/high 双档计费。

> 注意：**约束解码只保证格式合法，不保证内容正确**。schema 是必要条件而非充分条件，后面的 Gate 3/4 才是真正防线。

### 3.2 单图 token 与成本（1024×768 蜡烛图，官方规则推算）

| API | 计算规则 | 单图 token | 说明 |
|-----|---------|-----------|------|
| Claude | `(w × h) / 750` | **≈ 1049** | 官方公式，无缩放档位选择 |
| OpenAI | detail=high：`85 + 170 × tiles` | **≈ 765**（high）/ **≈ 85-110**（low） | low 档对 K 线细节（影线、副图）**大概率不够用** |
| Gemini | 图像固定 token | **≈ 1120** | 与分辨率无关，Flash 档单价最低 |

成本结构（估计，以实际牌价代入）：

```
方案A 每根 bar 调用:  96 次/日 × (1049 img + ~1500 prompt + ~600 out) token ≈ 30 万 token/日/单品种
方案B 信号驱动调用:   5~15 次/日 ≈ 3~5 万 token/日/单品种      → 约 6~8 倍成本差
```

多品种扩展时（BTC/SOL/XRP 等 6 品种）：方案 A 会到 ~180 万 token/日，方案 B 仍在 20~30 万以内。**这是选择方案 B 的第二个硬约束。**

### 3.3 本地部署可行性

- 框架：vLLM 支持列表包含 Qwen3-VL / Qwen2.5-VL / InternVL3.5 / GLM-4.5V / Molmo；SGLang 次之；Ollama 易用但吞吐与并发弱。
- 单卡 24GB（3090/4090）：8B 级 **FP16 勉强**（权重 17GB + KV + 视觉编码器高分辨率张量），实践上走 **AWQ-4bit / GGUF 量化**更稳。
- 吞吐：8B@4090 decode 数十至百 token/s，**单图 + 短输出端到端约 1~3 秒**（社区经验，需自测）。对 15m 收盘触发的链路完全够用（收盘后 1~3s 出信号不影响 15m 尺度执行）。
- 关键优势：**零边际成本 + 无外网依赖 + 可随意重跑历史**（回测时需要对过去几千个 bar 全量推理，这时 API 成本会成为不可接受的阻塞）。

### 3.4 选型决策（推荐两步走）

```
第一步（研究期，M4）：用第三方前沿 API 建立"能力上限基线"
  - 目的：回答"最好的模型在这批 K 线图上到底能拿多少分"，通过 Gate 1~4 才继续
  - 同时把它的标签输出存成标注数据集（免费得到蒸馏标签）
  - 回测期需要全量历史推理 → 用 Gemini Flash / GPT mini 级低价档 + 严格采样，或只在样本外窗口推理
第二步（生产期）：Qwen3-VL-8B(Apache-2.0) 本地 SFT/蒸馏
  - 用第一步的标签做 SFT（形态/状态分类是离散标签，天然适合蒸馏）
  - 推理走 vLLM + AWQ，零边际成本，全量历史可回测
  - 降级链：本地模型不可用 → 回退纯数值策略，绝不阻塞主链路
```

**风险警告**：小模型蒸馏后可能只学到"叙事模板"而非因果判别。FinAgent 的跨资产迁移失败、Backtrader-Bench 中"一半模型 ≈ 随机"是最直接的反例。因此本地 8B 上线前必须**重新完整跑一遍 Gate 1~4**，不能沿用 API 的验证结果。

---

## 4. 落地验证协议（Go / No-Go Gates）

这是本策略的核心资产。任何一道 Gate 不过，`vlm_v1` 只能停留在研究目录，不进信号链路。

### Gate 0 · 可审计数据前提（工程，一次性）

- [ ] point-in-time：VLM 只能看到 `ts_close <= t` 的 bar；渲染窗口不含未来
- [ ] 渲染幂等：同一 `bar_id` 只渲染/推理一次，产物落盘（`data/charts/`），记录 `prompt_version + model + temperature=0`
- [ ] 输出与时间戳严格绑定（`signal_id` 贯穿），杜绝"模型输出无法对应到某根 bar"这一 LLM 研究通病
- [ ] 知识截止检查：记录模型 cutoff，**cutoff 之后的区间单独切片评估**，若 cutoff 前表现远好于 cutoff 后 → 判定为记忆效应，直接淘汰

### Gate 1 · 增量能力测试（最关键，取代"绝对收益"评估）

```
对照: rule_v1 与 dl_v1 的样本外信号
实验: rule_v1 + VLM标签 → 净值/Sharpe/回撤/成本后收益
指标: Δ Sharpe、Δ 最大回撤、标签的 IC / AUC（对后续 4h/8h/16h 收益）
判据: 在 ≥ 2 年样本外区间上，VLM 标签对后续收益的 |IC| > 0.02 且
      加入后成本后收益不降、回撤不升；否则 no-go
关键: 必须同时跑"随机标签对照组"（permutation test），排除叙事运气
```

### Gate 2 · 时间视野敏感性（直接对应已知的失败模式）

```
同一张图，prompt 中分别要求预测未来 15m / 4h / 1d 的状态
判据: 三组输出分布必须有显著差异（如 χ² / 标签重叠率 < 0.7）
若输出几乎不变 → 模型未真正使用 horizon 信息，属于"时间绑定失败"，no-go（这是文献里最常见的死法）
```

### Gate 3 · 反事实与扰动鲁棒性

```
对同一行情做变换，标签应保持不变的必须不变，应翻转的必须翻转：
 a) y 轴缩放/平移（仅改视觉幅度，不改形态）→ 标签一致率 ≥ 85%
 b) 多空镜像（上下翻转价格）→ 多空标签必须对称翻转，否则说明存在方向偏置
 c) 加入 1% 高斯噪声到 OHLC → 一致率 ≥ 80%
 d) 移除副图（只留蜡烛）→ 性能衰减幅度可控，用于确认它是否真用了指标
 e) 空白/纯随机走势图 → 假阳性率（模型仍报出强形态）≤ 20%   ← 最能暴露幻觉
```

### Gate 4 · 置信度校准

```
把 confidence_raw 分 10 桶，统计每桶实际命中率 → reliability diagram
用温度缩放 / isotonic 在训练窗口拟合校准器（绝不在校准后的数据上评估）
判据: ECE ≤ 0.08；未校准的 confidence 严禁用于仓位缩放（否则 sizer 会被幻觉置信度放大风险）
```

### Gate 5 · 成本后严格回测

```
手续费/滑点 ×1 ×2 ×4 三档；资金费率逐笔结算；SL 优先假设；Monte-Carlo 交易重排回撤分位数
判据: ×2 成本后仍为正期望；回撤 P95 在 GlobalGuard 阈值内
```

### Gate 6 · 前向模拟交易

```
≥ 60 天（含至少一次趋势市 + 一次震荡市）；与 rule_v1 同区间并跑
判据: 实际前向表现与回测差距在预设容差内；飞书卡片记录的否决次数与理由可复盘
```

### 失败退路

Gate 1/2 任一不过 → **放弃图像路线**，改做：
- 把 OHLCV + 指标转成**结构化文本/表格**喂 LLM（可控、可回测、无视觉瓶颈）；或
- 走方案 C（LLM 生成因子表达式 + 确定性回测），这条路的可审计性最强；或
- 用纯视觉 CNN/ViT 做市场状态分类器（0.892 AUC 那条证据支持），它比 VLM 便宜几个数量级且无需 prompt 工程——但**同样必须对比数值基线**。

---

## 5. 训练路线评估（SFT / DPO / RL）

| 路线 | 适用条件 | 风险 | 建议 |
|------|---------|------|------|
| 提示工程（零训练） | 前沿 API，离散标签任务 | 成本高、不可本地化 | **M4 起点** |
| SFT 蒸馏（API 标签 → Qwen3-VL-8B） | 任务离散、标签量大、格式固定 | 学到模板而非因果；需重跑 Gate | 方案 B 的自然终点 |
| DPO / 偏好优化 | 有"好/坏决策"成对样本 | 成对样本极难构造（PnL 噪声大） | 暂缓 |
| RL（GRPO/PPO，奖励=成本后 PnL） | Trading-R1 类做法，4B 也可出结果 | **reward hacking**：模型学会高频刷单/只在有利时段出手；需 abstain 奖励 + 成本入奖励 + 严格时间切分 | 仅在 Gate 全过后作为 M6 研究项 |

补充：带弃权分支的选择性预测（FinAbstain / Reinforced Hesitation 一类）与本项目天然契合——`no-trade` 是合法且期望的输出，必须写进 schema，否则模型会被迫"编造理由"。

---

## 6. 对设计文档的修订项（需要落到代码时同步）

1. §6.4 `VLMStrategy`：输出 schema 去掉 `entry/stop/targets` 的价位字段，改为 `pattern / regime / setup_validity / veto / confidence_raw`；价位仍由 `risk/stops.py` 计算。
2. §6.4 触发条件：从"每 N 根采样"改为**信号驱动调用**（有候选 Signal 才评估）。
3. 新增 `strategy/vlm/gates.py` + `tests/test_vlm_gates.py`，实现 Gate 1~4；`vlm_v1` 默认状态 `research_only=true`。
4. 新增 `vlm/client.py` 支持 provider 抽象（`anthropic | openai | gemini | vllm`），预算超限自动降级为规则策略（原设计已含，需加本地 vLLM 分支）。
5. 新增标注数据集目录 `artifacts/vlm_labels/`（bar_id → 标签 + prompt_version + model），供蒸馏 SFT 复用。

---

## 7. 需要你确认

1. **"laya"** 具体指哪个工作/模型？（我未能检索到可验证来源，可能影响 §1 的取舍）
2. 是否有本地 GPU（型号/显存）？决定第二步能否走本地 8B（24GB 单卡可用 AWQ；16GB 建议 4B 档）。
3. API 预算上限（月）与允许的数据出境（本地部署可规避）。
4. 是否接受 `vlm_v1` 在 MVP 阶段仅为"否决器/标签器"而非独立发信号主体（我的建议：**接受**）。

---

## 参考来源

- [Do VLMs Truly "Read" Candlesticks? A Multi-Scale Evaluation](https://arxiv.org/abs/2604.12659)
- [Visual Chart Representations for Cryptocurrency Regime Classification](https://arxiv.org/abs/2605.00875)
- [FinChart-Bench: Evaluating Financial Chart Understanding of Large Vision-Language Models](https://arxiv.org/html/2507.14823v1)
- [FinAgent: A Multimodal Foundation Agent for Financial Trading](https://arxiv.org/html/2402.18485v3)
- [MarketSenseAI 2.0](https://arxiv.org/html/2502.00415v2)
- [MM-DREX: Multimodal Dynamic Routing Expert LLMs](https://arxiv.org/html/2509.05080v2)
- [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138) · [repo](https://github.com/tauricresearch/tradingagents)（社区报告时间泄漏：[issue#805](https://github.com/TauricResearch/TradingAgents/issues/805)）
- [FLAG-Trader (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.716.pdf) · [Trading-R1](https://arxiv.org/abs/2509.11420)
- [AlphaAgent: LLM Alpha Factor Discovery with Lookahead Detection (KDD25)](https://arxiv.org/html/2502.16789v2) · [Alpha-GPT 2.0](https://arxiv.org/pdf/2402.09746)
- [Are Language Models Actually Useful for Time Series Forecasting? (NeurIPS24)](https://arxiv.org/abs/2406.16964)
- [FINSABER: LLM in finance evaluation](https://www.ml-quant.com/papers/arxiv/2505.07078/) · [StockBench](https://stockbench.github.io/) · [Backtrader-Bench](https://arxiv.org/html/2608.11232v1)
- [The Alpha Illusion](https://arxiv.org/html/2605.16895v1) · [Profit Mirage / FactFin](https://www.alphaxiv.org/abs/2510.07920) · [Agentic Trading 综述](https://arxiv.org/html/2605.19337v1)
- 模型与部署：[Qwen3-VL](https://github.com/qwenlm/qwen3-vl)（[技术报告](https://arxiv.org/abs/2511.21631)、[8B 卡](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)、[吞吐基准](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)）· [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) · [InternVL3.5](https://arxiv.org/html/2508.18265v1) · [GLM-4.5V](https://arxiv.org/abs/2507.01006) · [Molmo](https://github.com/allenai/molmo)（[Molmo 2](https://allenai.org/blog/molmo2)）· [vLLM 支持模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html) · [vLLM 结构化输出](https://docs.vllm.ai/en/latest/features/structured_outputs/) · [vLLM vs Ollama vs SGLang 实践](https://northflank.com/blog/vllm-vs-ollama-and-how-to-run-them)
- API 计费等：[Anthropic vision 文档](https://platform.claude.com/docs/en/models/overview) · [Anthropic prompt caching 机制](https://technspire.com/en/blog/anthropic-prompt-caching-pricing-mechanics) · [OpenAI 定价](https://developers.openai.com/api/docs/pricing) · [Gemini 图像理解](https://ai.google.dev/gemini-api/docs/image-understanding) · [Gemini 定价](https://ai.google.dev/gemini-api/docs/pricing) · [GPT 图像 token 计算争议帖](https://community.openai.com/t/gpt-5-mini-image-input-token-calculation-discrepancy-with-official-faq-formula/1344040) · [约束解码有效性的社区讨论](https://news.ycombinator.com/item?id=46345333)

> 说明：部分 2026 年论文的结论仅基于摘要或作者自述，引用前建议复核正文；标为"估计"的性能数字需以你的实测为准。
