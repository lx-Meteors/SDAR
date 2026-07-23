# 研究方向决策记录

> **本文件已迁移：正本在 `/home/test/yyy/BEACON/research_decisions.md`**
> （2026-07-15 起，训练与 BEACON/GiGPO 三基线对比均在 BEACON 仓进行，
> 新记录只写那边；本文件保留为历史快照，不再更新。）

更新时间：2026-07-10

## 已排除的方向（不再考虑，请勿重新提出）

1. **belief-bin GiGPO**（对 logπ(gold|h_t) 做分位数分档作为分组键）
   - 已否决，相关提案文档 `belief_bin_gigpo_idea.md` 已删除。
   - 遗留的离线分析脚本（`ARPO/evaluation/ds_credit*.py`、`SDAR/tmp_scale/`）仅作为
     评测框架和历史数据保留，其中 belief-bin 分支不再作为候选方案。

2. **用 ARPO 分支机制"构造"锚点状态**（sibling 共享精确前缀 → 分支点即 step group）
   - 已否决，不考虑 Tree-GRPO / prefix-tree 式的 rollout 期构造碰撞方案。

3. **观测精确/模糊匹配**（GiGPO 原生 obs-anchor 及 SequenceMatcher 相似度）
   - 深搜场景已被实验证伪：bamboogle 16 题 × 8 rollout，obs 组 110/110 全 singleton，
     活跃组覆盖率 0%。

4. **规则化锚点**（字符串匹配 gold/桥接实体的进度状态、归一化查询碰撞、URL/doc-id
   集合指纹等一切手工规则）
   - 已否决：不接受规则化方案。方向要求：先保证有效，其次优雅；
     优先模型内部信号 / 学习型的锚点定义。

## 问题设定约束

- **目标任务是两个，同一方法要在两边同时成立**：
  1. **深度搜索**：`/data1/test/yyy/ARPO`（工作区内 `ARPO/`）中的深搜索题目，
     即 `ARPO/evaluation/data/` 下的多跳/搜索题库（2wiki、musique、bamboogle、
     hotpotqa、gaia、webwalker、xbench、SimpleQA、hle），math 类
     （aime24/25、gsm8k、math、math500）不在目标内；
  2. **WebShop**：`verl-agent` 的 GiGPO 训练设置，目标是同一方法打过 GiGPO。
- **奖励保持二值 0/1**，不用 F1 等连续奖励。
- **turn 是不可拆分的信用单元**：think + tool call 一起看，不在 turn 内部再切分。
- **目标聚焦长程信用分配**：把二值终局信号正确摊派到各 turn。

## 结构性结论（离线实验确认）

**二值终局奖励下的符号惰性定理**：只要 outcome R ∈ {0,1} 且 step 基线 b 是组内
outcome 的凸组合（GiGPO 硬分组均值、核加权软基线、可交换性加权、clip 到 [0,1] 的
值探针——全部属于此类），则 sign(R − b) 在非零处必然与 episode advantage 一致。
**任何配组方案都不可能在二值奖励下翻转符号**，差异只在"哪些步弃权"。
推论：二值奖励下长程信用分配退化为**轨迹内的量级分配问题**。

## 实验记录（bamboogle 16 题 × 8 rollout，脚本在 ARPO/evaluation/）

- `ds_hidden_anchor.py`：隐状态核软基线——长上下文表征近共线
  （跨 rollout cos≈0.95–0.99，same/diff-progress gap ≤0.006），塌缩为 episode advantage。
- `ds_exch_anchor.py`：二值奖励下可交换性锚点/值探针在与 episode 符号相悖的步上
  acc=0%（定理实证）。
- `ds_gate_f1.py`：F1 连续奖励下有真实符号翻转（可交换性 5 步 80% 正确），
  但 F1 路线随奖励约束作废。
- `ds_ablate_credit.py`（反事实 turn 消融信用，d_t = Δlogπ(自身答案|删 turn t)）：
  成功侧 AUC **0.85**；失败侧 0.13 反转（疑似"证据被误用"效应，n_good=5）；
  失败轨迹 blame 高度集中（top-1 turn 占 86%）。surprisal 加权两侧无效，已否决。
- `ds_opsd_pgrad.py`（OPSD 锚点 × 梯度词表重分配）：
  理论：PG 更新 Δz_v ∝ A_t·(1[v=y_t]−p_v)，蒸馏更新 Δz_v ∝ q_v−p_v，
  对齐两者的一阶最优优势调度 A_t ∝ q(y_t)−p(y_t)（OPSD gap）。
  结果：**OPSD 失败侧 blame 排序 AUC 0.74（首个正向失败侧信号）**，成功侧 0.41；
  与消融完全互补。OPSD 带符号做独立方向通道失败（flip-acc 13%）。
  一阶 dPhi 模拟：uniform −1 广播 mean=+0.407（68% 失败轨迹 margin 向 gold 移动），
  p-重分配通道使朴素负广播自带部分正确信用。
- 当前组合设计：方向 = episode 符号；成功侧量级 = 消融 credit（0.85）；
  失败侧量级 = OPSD blame（0.74）。全部模型内部、零规则、turn 整体。

## 实验记录：teacher 反事实动作锚点 + 正负相抵门控（2026-07-10 深夜）

**`ds_teacher_anchor.py`（GPU5）**：锚点键 = OPSD teacher 在 turn 前状态贪心解码的
反事实动作（π*-等价抽象的代理），组内 GiGPO 式优势。结果：
- 表面数字好看（多 rollout 覆盖 86%、sign-acc 81%）但**被 turn-0 主导**：
  turn-0 前状态 = 裸问题，跨 rollout 天然全同，任何确定性函数都平凡碰撞。
- 剔除 turn-0 后只剩 17 个深层 turn，覆盖率跌到 18%（3/17），
  且 **teacher 与 student 对照数字完全相同**——gold 条件化未提高动作碰撞率，
  "teacher 知识制造碰撞"的核心主张在本数据上未获支持。
- 定性上 teacher 有语义规范化能力（t0 处给出正确子问题分解，student 跑偏），
  但精确文本匹配无法在深层 turn 利用它。

**`ds_cancel_gate.py`（离线）**：跨 rollout 相同查询组量化 GRPO 正负相抵 + OPSD 门控。
- 相同动作组 11 个（覆盖 32/103 turn），**结局混合组只有 2 个**——本数据上
  相抵现象本身稀少，无法有效评测。
- 门控衰减方向正确但弱（失败侧好 turn 保留 37% vs 坏 turn 45%）；
  2 个混合组中修复 0、打翻 1（幸运成功轨迹未被 teacher 否决）。

**结构性瓶颈（最重要发现）**：bamboogle 3B 数据 86 条轨迹仅 103 个 turn
（平均 1.2 turn/轨迹，depth≥2 的 turn 仅 17 个）——**几乎全是单跳轨迹，
承载不了"长程"信用分配的任何评测**；此前全部 AUC 结论同受此限。
下一步必须先造深轨迹数据（更难多跳题 + 强制多步检索 + ≥16 rollout），
再重跑四套离线评测（ablate / opsd_pgrad / teacher_anchor / cancel_gate）。

## 深轨迹数据集（2026-07-11，解决结构性瓶颈）

生成配置：Qwen2.5-3B-Instruct，vLLM 双卡（6/7，端口 8006/8007），`search_deep`
提示词（强制原子子问题分解 + 每跳独立检索），`min_search_times=3` 提前作答拦截
（`src/sample_processor.py` 中的 nudge 机制），32 题 × 16 rollout，最多 16 turn。

| 数据集 | rollout | 方差题 | buried-good | 深层 turn | 混合动作组 | 平均检索/轨迹 | 正确率 |
|---|---|---|---|---|---|---|---|
| bamboogle（旧） | 8 | ~8/16 | 5 | 17 | 2 | 1.2 | — |
| **2wiki（主评测集）** | 16/16 完成 | 28/32 | **416** | **6760** | **62** | 3.5 | ~26% |
| musique（难档补充） | 生成中 | 7/32@7r | 55@7r | 663@7r | 8@7r | 1.9 | ~6% |

所有关键样本量比旧数据高 1–2 个数量级，此前"n 太小"的评测限制全部解除。
评测流水线：`ARPO/evaluation/run_deep_eval.sh <dataset>`（GPU5，四脚本按
ablate → opsd → teacher → cancel 依赖顺序串行，每个数据集独立工作目录
`eval_deep_<dataset>/`）。待验证三个数字：消融成功侧 AUC 0.85、OPSD 失败侧
AUC 0.74、cancel_gate 混合组行为（旧数据只有 2 个混合组，新数据 62 个）。

## 小验证结果（2026-07-11，2wiki 深轨迹 8 题 × 8 rollout，64 轨迹 / 260 turn）

方差题子集（q0–q8 中非平凡金答案且有成有败者），工作目录
`ARPO/evaluation/eval_deep_2wiki_small/`，全程 GPU5 约 13 分钟。

| 信号 | bamboogle（浅） | 2wiki 深轨迹小验证 | 结论 |
|---|---|---|---|
| 消融 d_t 成功侧 AUC | **0.85** | **0.53** | **阵亡：0.85 是浅轨迹伪影** |
| 消融 d_t 失败侧 AUC | 0.13 | 0.36（仍反向） | 阵亡 |
| OPSD(-signed) 失败侧 | 0.74 | 0.62 | 衰减但存活 |
| **lev_signed (p_o−p_g) 失败侧** | — | **0.68** | 深数据最佳失败侧信号 |
| **lev 杠杆质量 (p_g+p_o) 成功侧** | — | **0.69** | 深数据最佳成功侧信号 |
| 位置基线失败侧（诊断用） | — | 0.67 | 规则信号，不可用但需超越 |

其他三项：
- **一阶 dPhi（失败轨迹）**：uniform −1 广播 82% 轨迹 margin 向 gold 移动
  （mean +0.108）；levgate 最优（mean +0.156, 82%）——p-重分配结论在深数据复现。
- **teacher 反事实动作锚点**：剔除 t0 后深层覆盖 25%，teacher 66% vs student
  63% sign-acc——gold 条件化仍未提高碰撞率，与 bamboogle 结论一致，**方向搁置**。
- **cancel_gate**：混合结局组 9 个（旧数据仅 2）；OPSD 门控修复 1 打翻 0，
  好 turn 负优势保留 32% vs 坏 turn 52%（方向正确、幅度中等）。

**小验证综合结论**：深轨迹上唯一双侧存活的模型内部信号是 **OPSD 杠杆家族
（p_g / p_o 探针）**——成功侧用杠杆质量（0.69）、失败侧用带符号杠杆（0.68），
恰好取代阵亡的消融信用，且与"OPSD teacher 指导梯度重分配"的主线自然衔接。
下一步若做全量（32 题 × 16 rollout）评测，只需验证这两个数字是否稳住。

## Token 级三分类实验 + probe-not-guide 原则（2026-07-11 晚）

`ds_token_taxonomy.py`（2wiki 小子集 8 题 × 4 rollout，6089 个模型 token 位置，
GPU5）：对比 student p 与 gold 条件化 teacher q 的词表分布，位置分三类：
- **agree 74.2%**（TV<0.1）：脚手架 token，梯度应归零（降噪）；
- **leak 5.0%**（分歧落在 gold 词 + `</think>`/`<answer>` 收尾词）：teacher 的
  "抢答冲动"= OPSD 崩溃方向，**低维可过滤通道**（teacher argmax 前十：`</`、
  `think`、`answer`…）；
- **fork 20.8%**（分歧落在普通内容词）：真决策分叉，57% 在 think 段；
  高熵 top-10% 位置 71% 是 fork 但只覆盖 34%（分歧定位器严格强于 ARPO 熵信号）。

**关键实证**：fork 处 student 偏离 teacher argmax 的比例，成功轨迹 60% =
失败轨迹 60%——偏离不预测失败，**teacher 行为分布无资格做监督目标**
（信息集不同：teacher 条件在 gold 上不需要搜索，其最优动作测试时不可行）。

**probe-not-guide 原则（应对 OPSD 提前收敛崩溃）**：teacher 只允许三个只读角色
——探针（p_g/p_o → lev turn 量级）、定位器（分歧−leak → fork token 掩码）、
仲裁器（相抵门控）；不做行为目标，损失中无任何 KL(p‖q) 项。
token 级优势：A(t,i) = episode 符号 × lev(t) × fork 掩码(i)，leak 位置显式排除
（防崩溃机制的结构化实现）。student 保留：fork 处自己采样的 token 与探索熵、
检索行为、buried-good turn（负梯度被压小）。

## WebShop 统一方案（与深搜索同一方法打 GiGPO）

统一视角：GiGPO step 组均值 = 碰撞状态处的 MC 值估计；探针 p_g(h_t) =
免碰撞的模型内生值读数（二值奖励下成功 ≡ 最终发出 y*）。
- y* 统一定义 = **组内事后 gold**：深搜索为金答案；WebShop 为组内最高奖励
  rollout 的最终动作文本（商品+选项），零外部标签。
- 优势来源：覆盖率（singleton 全覆盖 vs GiGPO 深页面退化）、方差（确定性读数
  vs 2–3 样本 MC）、粒度（fork 掩码 vs step 内均匀）、超参跨任务通用。
- 训练前可证伪检验：碰撞状态处探针读数应与 GiGPO 组均值 outcome 正相关，
  singleton 处探针仍随进度变化。
- 集成点：`verl-agent/gigpo/core_gigpo.py::build_step_group` / 优势计算处
  drop-in 替换；conda 环境 `verl-agent-webshop` 现成。

## WebShop 训练前 mini 检验（2026-07-11，`webshop_probe_check/`）

规模：**3 任务 × 4 rollout × ≤8 步**，基座 Qwen2.5-3B-Instruct（temp 1.0），
GPU5 + vLLM 单进程（生成+探针），总耗时 ~2 分钟 rollout + 探针（冷启动 vLLM ~1min）。
脚本：`probe_check_mini.py`；产物：`mini_rollouts.json`、`mini_probe.json`、`mini2.log`。

| 判据 | 结果 | 判定 |
|---|---|---|
| A 锚点碰撞结构 | 36–72% 状态处跨 rollout 同 anchor | **通过**（WebShop 有碰撞，≠深搜索 singleton） |
| D 覆盖率 | GiGPO 激活 51% vs 探针 100% | **通过**（探针全覆盖仍成立，但优势小于深搜索） |
| C 轨迹内进度 | 高分轨迹 Δlogp_g(first→last)=+1.13(n=2)；零分 +0.23(n=6) | **弱通过**（探针随步数上升，方向对） |
| C margin | 高分 margin≈0，零分 margin=-0.75 | **弱通过**（高分轨迹更接近 y*） |
| B 碰撞处相关性 | spearman(probe_g, 组均值 score)=**-0.13** (n=21) | **未通过**（样本太小 + y* 定义粗糙） |

**关键发现**：
1. WebShop 与深搜索结构不同：GiGPO 在 WebShop 仍有 ~一半激活率，探针的"覆盖率优势"
   不如深搜索（0%）那么决定性；打 GiGPO 需要靠**方差/粒度**而非仅靠覆盖。
2. 当前 y* = 组内最高分 rollout 的**动作序列字符串**作探针目标，在 WebShop 上
   B 判据失败——可能需改为最终商品+选项文本，或用连续 task_score 而非二值 won。
3. 12 条轨迹、0 条 won=1.0（基座模型太弱），只能看 graded score 0/0.6/0.83；
   **不足以做训练前 go/no-go**，但足够排除"WebShop 完全无碰撞"的担心。

**快速迭代建议**（不重跑环境）：改 `mini_probe.json` 的 y* 定义 / 只读
`mini_rollouts.json` 做离线重算，秒级；要补 B 判据只需换 2 个任务 ID 或
用 GiGPO ckpt 替代基座（仍保持 3×4 规模）。

## Fork 候选菜单锚点实验（2026-07-11 夜，`ds_fork_anchor.py`）

主张：深搜索观测不重复但**决策重复**；候选 token 分布把观测空间投影成小菜单，
锚点应定义为 turn 级 fork 候选集（token id 集合，Jaccard≥0.4 成组），碰撞在菜单空间发生。
2wiki 小子集 8 题 × 8 rollout（260 turn，GPU5，~30 秒）：

| 锚点 | 深层 turn 碰撞覆盖 | 混合组 | 组内 hit 一致 vs 基线 |
|---|---|---|---|
| 观测精确匹配（GiGPO） | 53% → **剔除 nudge 假碰撞后 0%** | 16→0 | — |
| student 候选集 top-8 | 13% | 15 | 75% vs 56% |
| **fork 候选集（OPSD 定位）** | **19%** | **15** | **75% vs 56%** |

- GiGPO 表面覆盖全部来自注入的 nudge 文本假碰撞；真实深层覆盖 0%（singleton 再实锤）。
- **teacher 首次对锚点有实证增益**（19% vs 13%）：候选集合重叠比动作全文匹配
  （已否决的 teacher-action anchor）鲁棒得多，gold 条件化的语义归一化得以表达。
- 组内语义有效：例子组把 "release date"/"release year" 等不同表述归并为同一决策。

**当前信用分配总公式**：A(t,i) = sign(2R−1) × lev(t) × fork掩码(i) × 菜单组仲裁 g(t)。
分层覆盖：虚拟组（候选分布内 softmax 再分配，处处存在零方差）打底，
菜单锚点组（19% 深层 turn）提供跨轨迹对比，lev 配额 turn 量级。
长程性来源：所有探针目标都是终局 y*，中间决策在"距离终局"坐标系上度量。
与已否决方向的区分：非标量分箱（belief-bin）、非动作全文匹配（teacher-action）、
零字符串规则。

## 冗余审计 + 架构 v2：事后势函数（2026-07-11 深夜，`ds_potential_credit.py`）

**审计结论**：v1 四因子公式（sign × lev × fork × 菜单仲裁）有效但冗余——
三套机制五个阈值。所有信号实为同一原语的投影：**信念位移场 Δ_i = q_i − p_i**
（每位置全词表分布差）。lev=候选投影，fork=TV 范数，leak=gold/收尾词投影，
菜单锚点=支撑集重叠。

**新核心：事后势函数** Φ_t = logπ(y*|h_t)，turn 信用 = 望远镜差分
δ_t = Φ_t − Φ_{t−1}（Σδ_t = Φ_T − Φ_0 守恒；potential-based shaping 策略不变性；
处处有定义零方差）。与已否决 belief-bin 的区别：同一探针标量，但做轨迹内差分，
无分箱无分组。GiGPO 组均值 = 同一估计对象（V(s)）的碰撞依赖 MC 估计——
**探针是它的免碰撞零方差版本**（打 GiGPO 的估计量优势论证）。

2wiki 小子集验证（260 turn，GPU5 ~1 分钟）：

| turn 信用 | AUC fail(bad>good) | AUC succ(hit>miss) |
|---|---|---|
| **δ_gold（势差分）** | **0.75（新最佳）** | 0.58 |
| δ_margin(g−o) | 0.61 | 0.53 |
| lev 质量（候选投影，参照） | 0.42 | **0.69（仍最佳）** |
| lev_signed（参照） | 0.68 | 0.41 |
| 位置基线 | 0.65 | 0.53 |

失败轨迹 blame 集中度：top-1 turn 占 68% |δ| 质量。

**架构 v2**（机制从 3 套收敛为 1 个探针 + 投影）：
- 原语：事后探针对 (p_g, p_o)（y* = 深搜索金答案 / WebShop 组内最优终购）；
- turn 配额：失败侧 δ_t（0.75），成功侧 lev 质量（0.69）——同一探针的
  时间差分与位置敏感度两个导数，成功侧能否也统一到 δ 待后验；
- token 定位：fork 掩码（Δ 场 TV − leak 投影），leak 排除 = 防崩溃；
- 可选通道：菜单锚点组仲裁（仅在 19% 碰撞处激活，降级为增强项非核心）。
已淘汰：消融信用、teacher 贪心动作、belief-bin、菜单机制作为核心。

## 核锚点 + 成功侧变体实验（2026-07-12 凌晨，`ds_kernel_anchor.py`）

设计：无规则分布锚点——位置权重 w_i=TV(p_i,q_i)（连续、无阈值、无 leak 清单），
turn 嵌入 μ_t=Σw_i√p_i（Hellinger 核均值嵌入，零参数），核 k(s,t)=⟨μ_s,μ_t⟩，
James-Stein 收缩 Â_t=(1−λ)δ_t+λ·核加权邻居均值，λ=n_eff/(n_eff+1)
（singleton 自动退回自身 δ，无覆盖悬崖）。2wiki 小子集 260 turn：

| 变体 | AUC fail | AUC succ |
|---|---|---|
| δ_gold 原始 | **0.75** | 0.58 |
| 概率空间 dp / 相对增益 rel / 轨迹内秩 rank | 0.74/0.75/0.66 | 0.55/0.57/0.60 |
| **核收缩 shrunk** | 0.74 | 0.49 |

**两个否定结果**：
1. 成功侧 0.58 无法靠换测量空间修复（dp/rel/rank 全部 ≤0.60）——支持
   "oracle 失配"解释：多跳桥接 turn 有用但不含金答案串，δ 记功而 oracle 判 miss；
   成功侧真实上限需要更好的 oracle（2wiki/musique 自带分解标注可用）。
2. 零参数核过钝：n_eff 中位数 6.4、86% turn 有邻居（覆盖大增），但核加权
   hit 一致性仅 58% vs 基线 50%（硬 top-k 集合版是 75% vs 56%）——软化换来
   覆盖、丢了精度，收缩把好信号稀释（fail 0.75→0.74，succ 0.58→0.49）。
   **软硬两版锚点均对 δ 核心无增益：跨轨迹锚点通道正式降级为纯分析工具**，
   剩余增益空间不在锚点，在成功侧 oracle 与训练闭环。

## 修正边锚点 + 组中之组 payoff 检验（2026-07-12，`ds_override_anchor.py`）

用户指定方向：从 OPSD 定位 token 中选锚点（对比 KL/JS/熵差/override 指标），
词表分布找相似状态，恢复 GiGPO 组中之组。锚点键 = **修正边** (x,y) =
(student argmax, teacher argmax)，共享边即成组。2wiki 小子集结果：

| 选择指标 | 深层覆盖 | 混合组 | hit 一致 | **payoff AUC** |
|---|---|---|---|---|
| S1 override | **38%** | 15 | 55% | 0.34 |
| S2 override∧ΔH>0 | 17% | 13 | 62% | 0.41 |
| S3 override∧JS>中位 | 38% | 16 | 55% | 0.36 |
| S4 JS top10% | 23% | 14 | 59% | 0.30 |
| S2x（去 gold 复读边） | 10% | 13 | 63% | 0.44 |
| 菜单集合（补测） | 19% | — | 75% | 0.40 |
| 菜单+进度 |Φ差|≤0.5 | — | 10 | 0.44 |

payoff AUC = "buried-good turn 是否处于组均值更高的组"（组中之组给出正确信用
的必要条件），**全部变体 < 0.5（反向）**；组内 Φ 也反向（winners>losers 仅 0.34）；
剔除 turn0 后仍 0.38；组内相对 δ 对比 0.51（无信号）。

**结构性根因（关键负结果）**：深搜索 3B 的失败模式是**下游证据误用**
（拿到 gold 后在综合阶段丢失），不是上游搜索路径差。结局方差不由中间状态
解释 ⇒ 任何"状态分组 + 结局对比"的组中之组前提在此机制上失效——
锚点找得到（override 边 38% 覆盖），但组结局对比携带的是反向信用。
这是对方案空间的收敛：**跨轨迹结局对比通道正式关闭**（软/硬/边/进度约束
四种锚点 + 三种对比量全部试尽）；幸存通道 = 轨迹内势差分 δ（失败侧 0.75，
其 blame 恰好集中在下游误用 turn，与根因诊断自洽）。

## WebShop token 解剖 + 特权信息形态实验（2026-07-12，`webshop_probe_check/`）

新采 12 条全文 rollout（3 任务 × 4，`collect_ws_full.py`，含环境真 gold：
asin/名称/选项/价格）。`ws_token_anatomy.py` 对 16,428 个 token 位置做
p（无特权）vs q（结果 gold 特权）全词表对比：

1. **形态普查**：agree 71% / leak 1% / fork 27%。fork 集中在 think 区
   （占 think 的 30%），action 区仅 8%——决策在 think 里做，action 是抄写。
2. **指标对比（top-decile Jaccard）**：JS≈KL≈TV（0.71–0.86，可互换）；
   ΔH 与 JS 几乎不相交（0.19）且选中的是标点/虚词（the , . and）；
   surprisal 与所有指标不相交（≈0.10）且被强制格式的 `<th`（<think> 开头）
   淹没——**ΔH、surprisal 单独做定位器是陷阱，JS/KL 是正确定位器**。
3. **fork token 是什么（逐个看过）**：查询构词（shorts↔gym/workout）、
   商品选择（`[b`↔`[next`，真实动作分叉 JS=0.57）、属性槽（Price/Type、
   color/size）——全是任务内容词；leak 位置 q 推的是 asin 数字碎片
   （B/0/9/7）= 复读通道，必须屏蔽（与深搜索 leak 结论一致）。
4. **关键负结果：结果 gold 特权在 WebShop 近乎惰性**。在 gold 商品可点击
   的 21 个错误决策处，结果 gold teacher 的 margin 提升仅 +0.03（43% 提升率
   = 掷硬币）；Φ(结果 gold) 与结局 spearman 0.06，碰撞组 B 判据 −0.40。
   ——深搜索里最有效的"答案条件化"在执行型任务上失效。
5. **关键正结果（`ws_hindsight_teacher.py`）：过程后见之明特权有效**。
   teacher 条件 = 同组最优 sibling 的动作序列（组内免费可得，无需环境
   oracle）：margin 提升 +0.38，71% 决策处抬升 gold 动作，33% 大幅抬升，
   并在轨迹分岔的关键决策处直接翻转符号（−1.46→+1.42）。

**结构性结论：特权信息的正确形态随任务瓶颈变化**——
知识型（深搜索）：瓶颈是"不知道找什么" ⇒ 结果 gold（答案）条件化有效；
执行型（WebShop）：瓶颈是"此页面该点哪" ⇒ 过程 hindsight（成功 sibling
的动作轨迹）条件化有效。统一形式：q = π(·|h_t, ξ)，ξ = 组内已解出实例的
终端解（QA 取答案，执行任务取动作序列）——即 **hindsight 自蒸馏，组内
自给，零规则**。样本量警示：n=21 决策、3 任务，为方向性证据。

## 方案去规则化收敛（2026-07-12，设计层）

用户质询"特权信息怎么来 + margin 含义 + 多 token 答案"，方案收敛为零规则形态：

1. **ξ 的统一定义**：ξ = 组内回报更高轨迹本身（终端解）。0/1 EM 下成功轨迹
   的答案 ≡ gold（定义恒等），故"知识型取答案 / 执行型取动作序列"不是设计
   分支而是同一对象的投影；最纯版本整段轨迹入 teacher 上下文，用哪部分由
   模型涌现。ξ 与 reward 同源（gold 本来就是 reward 在用的标签），零新增监督。
2. **边界自动对齐**：全失败组无 ξ，恰好 GRPO advantage 也为 0——teacher
   存在域 = 组信号存在域，无需特判。argmax 选 sibling = reward-softmax
   混合 teacher 的 τ→0 极限，是 RL 原语不是规则。
3. **去掉最后两个阈值**：JS top-k → 连续加权（权重=JS 值）；leak 检测屏蔽 →
   **支撑集受限蒸馏**（在学生 p 的 top-p 支撑上重归一化后蒸馏，teacher 只能
   重排学生已考虑的候选，复读通道被机制封死而非被规则检测）。
4. **多 token 候选**：全部量序列级，链式法则把 margin 压缩到第一个分歧
   token（共享前缀贡献为零）——信用落点=分歧落点，无需指定；跨候选比较用
   每 token 平均 logp（长度归一）。margin 仅为诊断读数，训练以蒸馏方向进入。

最终形态：A_i = A_episode + β·JS(p_i,q_i)·(q̃_i − p̃_i)|_{supp(p)}，
q = π(·|h, ξ)，ξ = 组内更高回报轨迹。无字符串匹配、无阈值、无任务分支。

## T2 hindsight teacher 全解剖 + δ vs GiGPO 正面对比（2026-07-12，`ws_hindsight_anatomy.py`）

同一批 12 条 WebShop 轨迹上，把"有效的 T2 teacher（组内最优 sibling 动作序列条件化）"
补齐 token 解剖，并首次在带标签 step 上让 δ / token endorsement / GiGPO 三者正面对打
（step 标签 = 环境 gold 规则判定，仅作离线 oracle）。12,666 个 token 位置，45 个带标签 step。

**1. token 解剖（T2 下）**：agree 62% / replay-leak 5% / fork 33%（fork 占 think 区 37%、
action 区 6%）。指标结论在 T2 下完全复现：JS≈KL≈TV（Jaccard 0.74–0.83），
ΔH 与 JS 不相交（0.18）选中虚词（the , with），surprisal（0.10）被 `<th` 强制格式
淹没——**定位器只能用 JS/KL 族，ΔH/surprisal 陷阱结论跨 teacher 形态成立**。
fork token 实质：asin 消歧（`Q`→`QC`，teacher 推 sibling 买的商品）、策略词
（clicking→removing/narrowing）、查询改写（price lower than→under）。
**T2 新增两种 leak 形态**：prompt 自指涉（q 推 " expert" 复述特权文本）、
动作截断（q 推 `]</` 想把查询改成 expert 原文长度）——replay 通道比 T1 的
asin 碎片更弥散。

**2. 支撑集受限蒸馏在 WebShop 上被证伪（top-p 版）**：16 个"gold 商品可点却点错"
的决策处，gold 动作首分歧 token（`[b`，asin 开头）**100% 落在学生 top-p 0.9 支撑之外**
（rank 5–12），受限蒸馏把 T2 增益（raw 94% 决策抬升 gold）清零。
但 rank 全部 ≤12 ⇒ **支撑集应改为 top-k 候选菜单（k≈20）而非 top-p**：
留住菜单内重排能力，仍然机制性封死 wild replay token。方案收敛第 3 条据此修订。

**3. 长程信用正面对比（45 个带标签 step，34 好 / 11 坏）**：

| 信号 | AUC | 剔 t0 | 覆盖 |
|---|---|---|---|
| **δ（hindsight 势差分，y*=sibling 终购，纯 student 探针）** | **0.80** | **0.77** | 91% |
| GiGPO 锚点组优势 | 0.56（sign-acc 34%） | 0.68 | 76% |
| e_raw（fork JS × teacher 对采样 token 的抬压） | **0.19（反向）** | — | 67% |
| e_res（top-p 受限版） | 0.42 | — | 67% |
| z(δ)+z(e_res) 组合 | 0.72（不如 δ 单用） | — | — |

- δ 分任务 1.00 / 0.72 / 0.69，随 teacher 质量（1.0 / 0.02 / 0.33 分 sibling）优雅退化，
  弱 teacher 的 hindsight y*（买错的商品）仍给出 >0.5 的信用排序。
- **token 级 endorsement 方向反向（0.19）**：teacher 在好 step 上照样改写风格/复读，
  q−p 的符号不携带 step 好坏——WebShop 上再次实证 probe-not-guide：
  **token 通道只许做定位（JS fork 掩码）与量级，方向必须来自 δ/episode**。
  深搜索"fork 处偏离 teacher 不预测失败（60%=60%）"的同一结论的量化版。

**结论：δ 通道在 WebShop 上首次正面击败 GiGPO 组中之组**（0.80 vs 0.56，覆盖 91% vs 76%，
免碰撞零方差），且与深搜索共用同一公式（y* = 组内更高回报轨迹的终端解，
深搜索失败侧 0.75 / WebShop 0.80）。样本量警示：45 step / 3 任务 / 规则标签，
需 verl-agent 训练闭环验证。最终形态修订：
A_i = A_episode + δ_t 分摊 × JS fork 掩码，概率传递限制在 **student top-k 候选菜单**
支撑上；teacher 的 q−p 方向不进公式。

## 新主线：候选 token 影响场 + 守恒梯度重分配（2026-07-12 晚，`ds_candidate_value.py`）

用户指令：放下 δ 主线，聚焦（1）每个位置的候选 token 如何影响后续轨迹、OPSD 能否定位；
（2）token 级锚点/更细粒度组中之组；（3）守恒梯度重分配。
数学基础（`BEACON/recipe/SurprisalRedistribution - 副本.pptx` + 推广）：
GRPO 对 logit 的梯度 g_v ∝ A·π_y(1[v=y]−π_v)，和恒为 0；候选间按 π_v 比例分摊
是 log-softmax 求导副产品而非设计。替换为任意 ω_v（Σω=1）等价于损失
**L_i = −Â·π_y(1−π_y)·(log π_y − Σ_v ω_v log π_v)**（系数 detach），
采样 token 梯度严格不变，ω=p̃ 时精确还原 GRPO——重分配是零成本插拔的自由度。

2wiki 小子集（8 题 × 5 rollout，480 个探测位置 × ~6 候选，V_i(v)=Φ(prefix⊕v)
候选条件值探针，GPU5 约 5 分钟）：

**Q1 候选影响场——强阳性，OPSD 是正确定位器**：
- V-range：fork 1.53 ≫ agree 0.53 ≈ 高熵非 fork 0.40（3–4 倍）；
  spearman(JS, V-range)=**+0.62** vs spearman(熵, V-range)=+0.20——
  熵定位价值支点失败，JS/OPSD 成功（与 token 解剖的 ΔH/surprisal 陷阱结论自洽）。
- **首个正向的 teacher 方向信号（正确粒度）**：位置内候选排序
  spearman(q,V)=+0.25（70% 为正）vs spearman(p,V)=−0.09（41%）；
  限制在 student top-5 菜单内仍存活：q +0.19（62%）vs p +0.02（50%）。
  此前证伪的是"q−p 在采样 token 上做 step 方向"（AUC 0.19 反向）；
  **同一位置候选之间的相对排序 teacher 是懂的，student 自己的 p 完全不懂**。
- V0 与 V24（续写 24 token 后再探）spearman +0.32、78% 为正：瞬时探针是
  轨迹影响的有噪但方向正确的代理。fork 处采样 token 平均 regret 0.56。

**Q2 token 级菜单锚点——存在且比 turn 级健康**：fork 位置 top-5 候选集合精确
碰撞覆盖 25%（173 组，55 组选择分歧、87 组结局混合）；共享菜单的跨上下文
V 一致性 +0.50（70% 为正）——**可比状态存在，且在价值空间可比**；
结局混合组中 winner 的 token 在 loser 上下文里 V 也更高：83%（n=6，方向性）。
与已关闭的 turn 级结局对比不同：这里对比量是探针 V 而非组均值结局，
不踩"下游误用"根因。

**Q3 重分配——自由度真实，但天真 ω∝q 收益有限，且发现更大的靶子**：
- 失败侧（94 个 fork）：counter-mass 落在 V>V(y) 候选上的比例
  grpo 0.28 / opsd 0.33 / unif 0.31；限制 student 菜单后 opsd 0.30 vs grpo 0.28，
  E[V_ω−V_y] −0.43 vs −0.53——**方向对但幅度小**；原始 opsd 大幅优势
  （−0.14）是 leak 混淆（q 推的菜单外候选=gold 词，V 探针天然虚高），不可信。
- **成功侧关键发现：GRPO 的正梯度把 49% 的剥夺质量从严格更优（V>V_y）的
  候选身上抽走**——熵坍缩伤害的定量化：正样本更新在系统性摧毁更好的备选。
  重分配的主战场可能不在失败侧"还给谁"，而在成功侧"少抢谁"
  （side-asymmetric ω：失败侧向 q 倾斜、成功侧保护高 q/V 候选）。
- 落地形态：ω 限制在 student top-k 菜单（防 leak，机制性）、
  ω_v ∝ π_v·exp(η·q̃_v)，η 分侧取值，η=0 还原 GRPO。

**局限**：V 探针对 gold-content 候选有虚高偏置（leak 混淆，Q3 已按菜单限制校正，
Q1 的 +0.25 同理应以菜单内 +0.19 为准）；V0↔V24 一致性中等（0.32）；
Q2 advice 检验 n=6。下一步：（a）成功侧保护性重分配的一阶模拟；
（b）菜单锚点组内用 V 对比替代结局对比的完整评测；（c）WebShop 侧复刻
（teacher 换 hindsight sibling）。

## CCR v2：sibling 保护的守恒重分配（2026-07-12 深夜，`ds_ccr_sim.py`）

对上节两个方向的一阶验证，idea 收敛为 CCR v2：

**证伪（两条，勿绕回）**：
1. 成功侧 ω∝π·exp(−η·q̃) 全局倾斜无效：η 0→10 保护指标持平
   （better-mass kept 0.91 不动）——q̃ 排序（+0.19）太弱、单步流量太小。
2. 失败侧任何 teacher 倾斜是负优化：GRPO 按 π 分摊已把 60% 释放质量送给
   更优候选，q̃ 倾斜降到 0.50——**失败侧"还给谁"不需要修**。

**杀手级正结果（sibling-strip）**：token 级菜单碰撞处（覆盖 25% fork，
253 个正更新位置），成功轨迹的正梯度平均把 **46%**（中位 35%，p90 **99%**）
的 counter-mass 从另一条**成功 sibling 实际选过的 token** 上剥走；
剥失败 sibling 的只有 20%/6%——GRPO 正更新系统性优先摧毁被终局证明过的
替代选择（token 级正负相抵/兄弟残杀）。此病灶对一切 advantage 方法不可见
（GiGPO 只改标量优势，改不了 counter-mass 流向）。

**CCR v2 设计**：
- 干预：fork 位置 counter-mass 分摊，L_i = −A·π_y(1−π_y)(log π_y − Σω_v log π_v)；
- 成功侧（主战场）：碰撞处 ω_v ∝ π_v·exp(−η·1[v∈S+])，S+ = 成功 sibling
  在同菜单选过的 token 集；singleton 退回 GRPO；
- 失败侧：GRPO 原样；
- 长程性：S+ 由整条成功轨迹的终局背书（hindsight 路由），强于 teacher q̃
  的一次前向（q̃ 降级为定位/分析工具）；
- 系统效应：保护成功替代 = 组内多样性保持 = advantage 组不过早退化全同。
- 落地：collection 期 1 次 teacher 前向（JS fork 掩码）+ 组内 top-5 菜单
  集合运算（碰撞与 S+，零额外采样）；loss patch ~30 行。
  训练过程指标：熵轨迹、组存活率（非退化组比例）、pass@k。

## WebShop sibling-strip 复刻（2026-07-12 深夜，`ws_ccr_strip.py`）

12 条现有轨迹，菜单碰撞 + 选择分歧做决策点过滤（无需 teacher），~2 分钟。
（首次运行 share 在 p_y 极小处数值膨胀，已加去重/clip/p_y>0.95 过滤。）

| 统计 | 深搜索 | WebShop |
|---|---|---|
| 碰撞覆盖 | 25%（fork） | **32%（全位置）** |
| STRIP 正更新剥 sibling token | succ 46% / fail 20% | fail 46%（med 39, p90 99） |
| STRIP action 区 | — | **77%（中位 1.00）** |
| FEED 负更新喂最优 sibling token | — | **49%（action 区 88%）** |

1. 碰撞处 counter-mass 是组内成员选择的零和战场，action 区几乎 100% ——
   路由干预（ω）的杠杆在碰撞处最大化。
2. **失败侧不用修，跨任务二次确认**：负更新在碰撞处已自动把 49%（action 88%）
   释放质量喂给最优 sibling 的 token。
3. 数据缺口：每任务仅 1 条正优势轨迹，succ-succ 残杀未直接测得 ⇒ 
   **动态论断**：病灶随成功率增长出现（早期正更新剥失败备选=无害；成功率
   上升后 succ-succ 碰撞对出现=残杀开始），恰好对应 GRPO 后期熵坍缩/平台期。
   训练中可用"succ-succ 碰撞对数 × strip 份额"曲线直接验证。
   训练前可用 GiGPO ckpt 重采 12 条 rollout（成功率高）先实测。

## succ-succ 残杀实测（2026-07-12 夜，GiGPO/BEACON ckpt 重采）

BEACON 1.5B step150 ckpt（`scripts/model_merger.py` 合并到 `.../step150_hf`）重采
同 3 任务 × 4 rollout（`ws_collect_ckpt.py`，注意 conda env 需 PATH+JAVA_HOME
指向 verl-agent-webshop 环境内的 openjdk）。分数：task0 全 0 / task1
[1.0,0.8,1.0,1.0] / task2 全 1.0。`ws_ccr_strip.py` 改为按 sibling 优势符号分桶。

| 统计 | 基座 | 训练后 ckpt |
|---|---|---|
| 碰撞覆盖 | 32% | **69%** |
| STRIP 剥正优势 sibling（残杀） | 无对可测 | **mean 0.79 / med 0.98 / p90 1.00**（n=100） |
| STRIP 剥非正优势 sibling | 0.46 | 0.72 |
| advantage 全零死组 | 0/3 | **2/3** |

1. **残杀在收敛策略上接近全额**（中位 98%）：同批次两个被正向强化的成功
   rollout 在碰撞决策点互相剥夺对方的 token——动态论断实证：46%（基座，
   剥失败备选，无害）→ 98%（训练后，剥成功备选，内耗）。
2. 组死亡（2/3 任务 advantage 全零）与残杀同框——CCR 保护成功备选 =
   保持组内多样性 = 延缓组死亡，两个病灶一个干预点。
3. 样本量警示：残杀对全部来自 task1（唯一有方差组）；graded 优势差小
   （±0.05/−0.15），但 strip 份额只量分摊结构、与优势大小无关。

**结论：CCR v2 的训练前证据链闭合**（病灶存在 → 随成功率增长 → 收敛时
接近全额 → 干预点唯一且杠杆最大）。下一步：verl-agent loss patch + 训练闭环，
过程指标盯 succ-succ strip 份额、组存活率、熵、pass@k。

## CCR v3：全词表场形式（2026-07-13 凌晨，用户去规则化质询）

用户否决集合匹配（S+ 菜单碰撞版不优雅），要求全词表、OPSD 定位、候选级指标。
收敛为场形式（`ws_ccr_field.py` + 候选信号对比）：

**ω_v ∝ p_v^{1+γ}·q_v^{−γ}（clip log(q/p) 至 ±5），v≠y，仅成功侧**，
q = π(·|h, ξ)，ξ = 组内最优 sibling **全轨迹**（非动作序列）。
= p·exp(−γ·候选惊讶度差)，即 log-ratio 指数族倾斜。

四个自动性质：自门控（agree 处 q≈p → 还原 GRPO，fork 定位涌现、无掩码）；
软支撑（p^{1+γ} 因子，无 top-k）；leak 反转为盟友（复读冲动在防剥夺方向
正确，失败侧保持 GRPO 故无喂质量危险）；S+ 集合版是其硬极限
（q 天然抬 sibling 成功 token）。

**候选级信号对比**（480 位置 V 探针）：q +0.25/菜单内 +0.19 最佳；
log(q/p) +0.22/+0.17 次之；q−p +0.16/+0.08；**p −0.09 反向——GRPO 的
按 p 分摊是所有选项里唯一系统性负向的分摊器**。ΔH/熵无候选级版本（不用）。

**场版打实测残杀对**（ckpt 72 对，GRPO 基线 0.76/med 0.95）：
- teacher=动作序列：无效（0.76→0.76）——残杀多在 think 措辞分叉，
  动作特权不含 sibling 措辞 ⇒ **特权粒度必须匹配残杀发生地，全轨迹必需**；
- teacher=sibling 全轨迹：γ=1 → 0.64/0.87，γ=2 → **0.54/0.70**；
  个例 ' query' 0.40→0.01（q_yb=0.67 认出 sibling 措辞）；
- **结构性残余**：二元位置（仅两个真实候选）无处路由，守恒下限，
  任何位置内方案（含集合版）都保不住；保护只在 ≥3 真实候选处生效。

对旧结论的解释：此前 exp(−η·q̃) 无效是因为 q 概率值太小（exp(−η·0.02)≈1），
log-ratio 尺度自由 ⇒ 换成惊讶度差后倾斜真实起效。

## teacher 几何对比：前缀 vs turn 局部注入（2026-07-13 上午，`ds_local_teacher.py`）

用户假设：hindsight 注入上一 turn（伪造检索结果/决策点旁备注）做最小 do-干预，
可能优于前缀特权（信息集混淆 + 注意力稀释 + 格式外通道三个论点）。实测否定：

| 指标 | 前缀 | turn 局部 |
|---|---|---|
| spearman(q,V) @fork / 菜单内 | +0.25 / +0.19 | +0.26 / +0.20（持平） |
| 配对胜率 | — | 38%（不胜） |
| **agree 位置污染 \|q(y)−p(y)\|** | **0.026** | **0.082（3 倍恶化）** |
| 自门控比 fork/agree | **15.0×** | 4.4× |
| WebShop 残杀保护 γ=2 | 0.54/0.70 | 0.57/0.72（持平略差） |

解释：候选间**相对排序**对 teacher 几何不敏感（注意力稀释只影响幅度；
信息集混淆只污染 q 作为价值/行为目标的用法——回溯解释了为何 q 排序独活）；
近因效应无差别放大所有位置扰动，直接损伤自门控。
**裁决：保留前缀几何 + sibling 全轨迹内容；turn 局部注入记为已测负结果。**
新增机制健康度指标：**自门控比（fork/agree 扰动比，前缀=15×）**，
今后任何 teacher 形态改动必须复测此数（防 OPSD 崩溃的量化哨兵）。

## CCR v4：几何桥双向族 + 两个优雅性质询（2026-07-13 上午）

用户两点质询：① sibling 全轨迹放 prompt 不优雅、"最优 sibling"如何判定；
② 内层重分配应双向（teacher 好→靠近，坏→远离）。

**① 的回答**：
- "选哪个 sibling"不是规则：q̄ = Σ_j softmax(R_j/τ)·π(·|h,ξ_j) 的 τ→0 极限
  = argmax-R；tie 时原则解为均匀混合，工程等价为每次 collection 均匀采样一条
  （期望无偏、零超参）。问题只剩 ξ 长度。
- **短 ξ 实测失败**：ξ = 终端解（最终购买 asin+options，几十词，从 sibling
  动作自动重构、零外部标注）在 72 残杀对上 0.76→0.76/0.71（γ=1/2），
  与动作序列特权同样无效。**结论三连**：outcome-gold 惰性 → 动作序列无效 →
  终端解无效；残杀发生在 think 措辞分叉，认出"sibling 的措辞也通向成功"
  所需的信息只存在于 sibling 的过程文本中。**全轨迹是过程 hindsight 的
  信息下界，不是工程偷懒**；代价框架化：每组成员一次前缀 forward（KV 缓存
  复用到全部位置）。
- 深搜索无此问题（ξ = gold 答案本来就短）。

**② 双向统一族（几何桥）**：

    ω_v ∝ p_v^{1−β} · q_v^{β}，β 由优势符号定向

- 成功侧 β=−γ_s（γ_s≈1~2）：= v3 原式。好候选少被剥（护）、坏候选多被剥
  （斥）——双向已内含于剥夺流的相对分配；
- 失败侧 β=+γ_f（γ_f≈0.5）：ω ∝ √(p·q)（Bhattacharyya 配权）。释放质量
  多喂好候选（引）、少喂坏候选（斥）。leak 由 p^{1−γ_f} 因子机制性压制
  （γ_f<1 必要；γ_s>1 允许，因剥夺方向 leak 是盟友）。
- 几何图像：ω 在 p、q 之间的测地线上，优势符号决定朝 q（喂）还是背 q（剥）。

**失败侧 log-ratio 补测**（此前只测过 q 概率值倾斜=尺度坏，94 fork）：
mass-on-better 0.28→0.29(β=.5)→0.31(β=1)；E[V_ω−V_y] −0.524→−0.484→−0.386；
菜单外质量（leak 代理）0.00→0.04→0.20。**判读：失败侧引力真实但弱**，
β=0.5 是 leak 安全点；β=1（纯 ω∝q）V 增益最大但 leak 质量 20% 不可接受。
成功侧在深搜索 V 探针上中性（0.49→0.51，噪声内）——成功侧的证据仍以
WebShop 实测残杀保护（0.54/0.70）为准，V 探针的聚合排序度量不敏感于
"保护特定成功 token"这一目标。

**当前裁决**：主战场维持成功侧（γ_s=1~2）；失败侧 γ_f=0.5 作为温和附加项
带入训练消融（默认开，效果弱但方向一致、leak 安全）。

## 通用特权形式：hindsight 充分性阶梯（2026-07-13 上午，`ds_sib_teacher.py`）

用户三问：① WebShop 能否也放 gold answer；② 答案 logp 是否适合做 hindsight
信号；③ 只做成功侧。

**② 理论打通（Bayes 翻转）**：候选级答案 logp 提升
ΔΦ(v) = log π(ξ|h⊕v) − log π(ξ|h) 与 prompt teacher 的 log(q_v/p_v)
由 π(v,ξ|h) 的两种分解相等（模因果 LM 的顺序近似）。
即：**log(q/p) 场 = 答案 logp 探针的一次 forward 全词表版**；
实测 spearman(log(q/p), V)=+0.22 证实近似成立（V 本身就是答案 logp 探针）。
分工：log(q/p) 做训练时全词表场（一次 forward），逐候选 logp 探针只做
离线审计（每候选一次 forward，做不了全词表）。

**① 实测（456 配对位置，同位置同 V）**：把深搜索 teacher 的 ξ 从 gold 答案
换成成功 sibling 全轨迹（含 tie 均匀混合版）：

| teacher | spearman(q,V)@fork | 菜单内 | 自门控比 |
|---|---|---|---|
| gold 答案 | **+0.24** | +0.17 | **14.8×** |
| sibling 全轨迹 | +0.17 | +0.14 | **2.8×（污染 agree=0.177）** |
| sibling 混合（≤3条平均） | +0.17 | +0.14 | 2.8× |

**全轨迹在深搜索上反而更差**：同题全文的复读冲动污染 agree 位置
（0.026→0.177），自门控哨兵 14.8×→2.8×，排序也降。与 WebShop 方向相反
（那边短 ξ 三连阴性、全轨迹必需）。

**统一解释：ξ 应取"成功的最小充分统计量"**。知识任务中答案语义上决定
过程 ⇒ 答案已充分，加轨迹只加复读噪声；执行任务中结果不决定过程
（哪些中间措辞通向成功的信息只在过程文本里）⇒ 全轨迹是信息下界。

**通用形式（零外部标注，双任务同构）**：
ω_v ∝ p^{1+γ}·q^{−γ}，仅成功侧（成功侧 only ⇒ 组内必有成功 sibling ⇒
特权永远组内自举，gold 标签在深搜索退化为"成功 sibling 的最终答案"，
内容相同但来源是组内经验）。ξ 沿**充分性阶梯**取最短充分档：
答案 → 答案+动作序列 → 全轨迹；判据用两个已建立的哨兵：
自门控比（≥10× 级）+ 残杀对保护实测。深搜索停在第一档，WebShop 走到第三档。
失败侧裁决：**不做**（用户拍板；此前实测引力真实但弱、且有 leak 风险，
放弃后 leak-as-ally 性质完整保留）。

## CCR v5：OPSD 门控摊平——保护不需要方向（2026-07-13 中午，用户否决全轨迹）

用户否决 sibling 全轨迹特权，提议对照式（正确+错误答案都放入）。实测 +
一个关键反思，得到比全轨迹更强且只需短特权的最终形式。

**对照式特权实测**：
- 深搜索（480 配对位置）：gold+3 错误答案排序 +0.25→+0.27，但自门控
  15×→7.1×；gold+1 错误 +0.23 / 12.2×。轻微得不偿失，不采用；
- WebShop（72 残杀对）：成功终端+失败终端对照，无效（0.76→0.76/0.75）。

**关键个例揭示本质**：同一紧凑特权下 ' with' 被护（q=0.84>p=0.44，
0.78→0.34），' query' 被反向加害（q=0.07<p=0.26，0.40→0.64）——
**措辞分叉处任何紧凑 hindsight 的方向都是噪声**（措辞×成功的耦合信息
只在过程文本里，信息论上无紧凑表示；全轨迹有效的本质是逐字复读）。

**反思：保护不需要认出"谁好"，只需要知道"这里 hindsight 无差别"**。
softmax 对 logit 整体平移不变 ⇒ 把剥夺质量均匀摊到全词表 = 纯粹抬高 y 的
相对 logit，候选间相对结构不动，残杀按构造为零，无需任何 sibling 信息。
GRPO 病灶 = ω∝p 把剥夺砸在最高 p 的替代项上（恰是 sibling 最可能采样的）。
teacher 角色从"指方向"退到"定位"：OPSD 散度标出"成功信息与此位置有相互
作用"的 token，在那里摊平。

**CCR v5 统一式**：ω_v ∝ p_v^{1−λ_i} · (q_v/p_v)^{−γ}，
λ_i = min(1, TV_i/c)，q 用**短 outcome 特权**（深搜索=答案；WebShop=
组内成功 sibling 终端购买，零外部标注）。仅成功侧。

**实测（72 残杀对，短终端特权）**：
- 门控本底：全部位置 TV med=0.000，仅 11% 超 0.1（89% 位置门关死=GRPO）；
  残杀对位置 TV med=0.124（门开）——OPSD 定位与残杀位置天然重合；
- 保护：c=0.1 → **0.76→0.21，中位数 0.95→0.00**（全轨迹方向版最好才
  0.54/0.70）；c=0.2 → 0.37/0.05；残余 0.21 来自 TV<0.03 的 10% 低散度对；
- γ 项（方向）在 WebShop 上不增不减（噪声对消），深搜索上留用（排序 +0.25），
  作训练消融旋钮。

**结构优势对比 v4**：特权重新变短且双任务同构（答案/终端解）；
不需要"哪条 sibling"（终端解 tie 时内容相同）；二元位置结构性残余也解决
（摊平不需要第三候选可路由——质量摊到全词表）；保护上限更高。

## v5 新鲜数据验证 + 候选级方向指标全对比（2026-07-13 中午）

**① 新鲜 rollout 验证**（`ws_collect_ckpt2/3.py` + `ws_ccr_v5_fresh.py`，
20 个新任务 × 4 rollouts，方法设计从未见过这些数据）：
- 组分布：11/20 组饱和或全等分（dead，无更新）、5 组仅一个正优势成员，
  **4 个方差组产生 221 个 succ-succ 残杀对**（含低分组 0.43/0.5 的
  相对成功者，非只有 1.0）；
- 门控本底复现：全部位置 TV med=0.000、仅 12%>0.1；残杀位 TV med=0.091；
- **保护复现且更好：GRPO 0.60/med 0.74 → v5(c=0.1) 0.15/med 0.00**；
  c=0.2 → 0.29/0.09。结论：v5 摊平在从未见过的任务分布上成立。

**② 候选级更细指标全对比**（用户问：logp 比值/熵差/其他能否分出
"哪个候选 teacher 更懂"）：
- 深搜索（240 fork，V 为真值）：单 gold log(q/p) +0.217 / 配对方向准确率
  60.0%；三特权集成均值 +0.221 / 60.1%；集成 SNR +0.199 / 58.9%；
  排名位移（scale-free）+0.188；符号一致门控 +0.214（覆盖 89%）。
  **全部与单 gold 持平或更差——单特权 log(q/p) 已在信息上限**；
- WebShop（72 已知真值对，y_b 确定该被保护）：三种短特权的方向集成，
  mean 方向>0 仅 50%（=掷硬币）；即使三特权符号一致（54% 的对），
  方向正确率也只有 56%。**任何候选级细化都无法从紧凑特权中提取
  措辞分叉的方向信息——信息本身不存在，非指标不够好**；
- 熵差无候选级版本（熵是分布性质，候选级对应物=排名位移，已测，更差）。

**裁决**：候选级不再细化。方向项只保留 log(q/p) 一个（深搜索有效侧），
WebShop 侧由位置级 TV 门控 + 摊平独立完成保护。v5 形式冻结，
进入训练实现阶段。

## 门形状对比：去手工拐点，单参 vs 自适应（2026-07-13 下午，`ws_ccr_v5_fresh.py`）

用户质询 λ=min(1,TV/c) 的硬拐点，问 TV 定义、能否单参、自适应 vs 预设。
7 种门在 221 新鲜残杀对上正面对比（保护=sibling token 被剥份额；
优雅=安静位 TV<0.02 处的平均 λ，越接近 0 越好）：

| 门 | 参数 | 剥夺 mean/med | 安静位λ | λ>.5占比 |
|---|---|---|---|---|
| 硬 c=.1（原） | 1 | 0.15/0.00 | 0.016 | 21.7% |
| 硬 c=.05 | 1 | 0.06/0.00 | 0.032 | 28.4% |
| **平滑 TV/(TV+c) c=.03** | 1 | **0.09/0.00** | 0.040 | 27.1% |
| 1−exp(−TV/c) c=.05 | 1 | 0.10/0.00 | 0.029 | 25.7% |
| λ=TV（零参） | 0 | **0.55/0.61** | 0.002 | 0.9% |
| λ=√TV（零参） | 0 | 0.44/0.36 | 0.021 | 3.4% |
| **自适应 min(1,TV/q90)** | 1 | 0.18/0.00 | 0.023 | 19.9% |

（GRPO 基线剥夺 0.60/0.74。）

**TV 定义**：TV_i=½·Σ_v|p_v−q_v|，全词表；有界[0,1]、对称、尾部鲁棒
（KL 会爆）、量纲与 ω 同为"质量份额"。

**结论**：
- 零参不可行：λ=TV 剥夺 0.55≈GRPO 0.60（残杀位 TV 中位仅 0.091，
  摊平强度被 teacher 扰动上界锁死）。"零参+有效"不可兼得；
- 平滑 TV/(TV+c) 去硬拐点且保护更好（0.09<0.15），c=半开点 TV 值，
  处处可导 → **定为默认**；
- 自适应 q90：参数从绝对尺度 c 变无量纲百分位 ρ，保护持平（0.18/0.00），
  机制优势=训练中策略锐化/teacher 漂移时自动跟踪 TV 尺度（离线测不出，
  待训练验证）；失效模式=无分叉步 q90≈0 放大噪声，需 max(q90, floor) 兜底
  → **定为训练鲁棒版，与默认做 A/B**；
- γ 方向项降级为可选扩展（WebShop 噪声/深搜索有用），主方法=纯摊平+单参门。

v5 门形式收敛：默认 λ=TV/(TV+c)（c≈0.03），训练期 A/B 自适应百分位版。

## 真实梯度 A/B：v5 vs GRPO 打进参数（2026-07-13 下午，`ws_v5_trainstep.py`）

单参数最终形式：**ω_v ∝ p_v^{1−λ}，λ = TV/(TV+c)，c=0.03，仅成功侧**，
其余照旧。代理损失 L = −A·(1−π_y).detach()·(z_y − Σ_{v≠y} ω_v.detach()·z_v)
（GRPO 情形 ω=π_v/(1−π_y) 时精确还原标准策略梯度）。

两个方差组（30 条成功侧序列、157 残杀位）、3 步 Adam lr=2e-5，
teacher 每步重算（与真实训练一致）：

| | GRPO | v5 |
|---|---|---|
| p(y_a) 自学 | +0.007 | **+0.011** |
| p(y_b) sibling | **−0.022** | −0.014（伤害少 36%） |

**判读**：首次在参数空间验证方向正确——学习不受损、sibling 直接伤害
显著减小。v5 下 p(y_b) 仍降是参数耦合的间接效应（其他位置更新经共享
参数移动一切分布），任何位置级方案都挡不住，v5 挡的是直接项
（离线份额 0.60→0.15）。Adam 首步瞬态大（GRPO ep0 p(y_a) 0.46→0.36
反弹），熵读数被污染，不作依据。

**离线证据链至此榨干**。剩余唯一问题（保护的多样性能否在采样-更新
反馈循环中兑现为方差组存活→成功率）只能由真实训练对照回答：
GiGPO vs GiGPO+v5，各 150 step，盯熵/死组比例/pass@k/成功率。
脚本支持 SAVE=1 存更新模型，可选做行为 rollout 对比。

## λ 定稿：自校准 odds 门（2026-07-13 下午，用户质询 c=0.03 刻意）

理论重构：λ 的本质 = 后验 P(真分叉 | TV)。TV/(TV+c) 恰是双峰分布等先验
下的后验 odds 形式——函数形状不刻意，c 只是背景噪声尺度，而噪声尺度
可以从数据估计：**c = 批内 TV 均值**（双峰分布的均值必然落在两峰之间，
实测全位置均值 0.034 ≈ 手调 0.03，非巧合）。

八种门实测（221 新鲜残杀对；保护=剥夺份额，GRPO 基线 0.60/0.74；
优雅=安静位 TV<0.02 平均 λ，应≈0）：

| 门 | 剥夺 mean/med | 安静位λ | λ>.5占比 |
|---|---|---|---|
| smooth c=.01 | 0.04/0.00 | 0.083 | 34.4% |
| smooth c=.03 | 0.09/0.00 | 0.040 | 27.1% |
| smooth c=.05 | 0.17/0.03 | 0.026 | 21.7% |
| smooth c=.1 | 0.31/0.13 | 0.014 | 12.0% |
| smooth c=.2 | 0.43/0.31 | 0.008 | 4.9% |
| **self c=meanTV** | **0.14/0.01** | **0.034** | 24.1% |
| self c=q90 | 0.18/0.00 | 0.023 | 19.9% |
| 1−BC（无参） | 0.60/0.74 | 0.000 | 0.1% |

**结论**：
- c 是敏感的（0.01→0.2 保护 0.04→0.43 单调滑坡），用户提的 c=0.1
  保护缩水一半（0.31）——**定值 c 不可取，自适应必要**；
- **自校准 self c=meanTV 定稿**：保护 0.14/0.01 与手调最优区间持平，
  零预设参数（c=当前批统计量），训练中自动跟踪 TV 尺度漂移，
  贝叶斯解释完整（λ=后验，噪声尺度由数据估计）；
- 最后一个无参候选 1−BC 也死了（≈TV²/2，量级更小，剥夺 0.60=GRPO 无保护）。
  **"无参数必然保护不足"结论闭合**：TV≈0.1 的证据幅度撑不起 λ≈1 的
  干预幅度，中间必须有一次由数据（而非人）校准的放大。

**v5 最终冻结版**：ω_v ∝ p_v^{1−λ_i}，λ_i = TV_i/(TV_i + mean_batch(TV))，
仅成功侧，teacher=组内成功 sibling 短 outcome 前缀。人为预设参数：0 个。

## 校准作用域：per-group 定稿（2026-07-13 下午，`ws_gate_scope.py`）

用户问 mean(TV) 能否按 group 算。三种作用域实测（221 对，GRPO 0.60/0.74）：

| 作用域 | 剥夺 mean/med | 安静位λ |
|---|---|---|
| per-step | 0.14/0.01 | 0.034 |
| **per-group** | **0.13/0.01** | 0.033 |
| 全局 | 0.13/0.01 | 0.032 |

组间 meanTV：0.034~0.042（本批任务同质，作用域不敏感——保护三者持平）。

**裁决：per-group 定稿**，理由是概念与鲁棒性而非本批数字：
- 概念对齐：残杀是组内现象（sibling 互抢），"该位置是否异常敏感"的参照系
  应为本组自己的散度地形；
- 任务异质性归一化：teacher 信息量因任务而异，per-group 把这层方差
  归一（全局均值会让弱信息组的分叉够不着门槛）；本批组间只差 1.2×
  所以测不出差别，异质任务分布下会显现；
- 统计量：一组约 3000 位置，均值稳（比 per-step 稳 4~5 倍）；
- 实现自然：GRPO/GiGPO 批本来按组组织；
- 退化组补丁：floor=0.005 兜底；训练版可升级为层级收缩
  c_g=(n_g·组均值+n₀·EMA)/(n_g+n₀)，同时吃小样本噪声与训练期漂移。

**v5 训练版最终形式**：ω_v ∝ p_v^{1−λ}，λ = TV/(TV + max(mean_group(TV), floor))，
仅成功侧，teacher=组内成功 sibling 短 outcome。预设参数 0（floor 为数值兜底）。

## 门定稿 v2：等权混合消掉 floor（2026-07-13 下午，`ws_gate_null.py`）

用户质询 max(mean_group, floor) 的硬钳位。分析：floor 防退化组
（teacher 无信息 → m_g→0 → λ=噪声/平均噪声随机开门），但退化组的
正确行为是 λ→0 回退 GRPO——floor 只压爆炸不实现回退，缺的是先验尺度。
层级收缩 c_g=(n_g·m_g+n₀·m̄)/(n_g+n₀) 引入伪计数 n₀；取 **n₀=n_g（等权）**
后伪计数消失：

**λ_i = TV_i / (TV_i + (m_g + m̄)/2)**，m̄ = 训练期批均值 TV 的 EMA。

退化组时门槛自动落到 m̄/2——floor 从结构里长出来，不再是外加常数。

**退化 teacher 实测**（组 7 换无信息特权，1892 位置）：
- 无信息前缀的扰动本底真实存在：meanTV=0.0125、p90=0.047
  （任何前缀都经注意力稀释扰动分布，"零 TV"不成立）；
- 假开门率（λ>0.5）：纯组均值+floor **23.8%** → 等权混合 **17.8%**
  （降 1/4）；mean λ 0.209→0.162；
- 信息组不受影响：c_pure=0.0377 vs c_blend=0.0382（比 1.01，保护不变）；
- 残余假开门的伤害有界：摊平无方向，假开只损失该位置的锐化，不会指错方向。

**最终冻结（真最终）**：ω_v ∝ p_v^{1−λ}，λ = TV/(TV + (m_g + m̄)/2)，
仅成功侧，teacher=组内成功 sibling 短 outcome。预设常数 0 个
（m_g=组统计量，m̄=运行统计量）。

## 门定稿 v3：用户裁决纯组均值（2026-07-13 傍晚）

用户否决混合形式，定 **λ = TV/(TV + mean_group(TV))**，无 floor、无 m̄。
数据支持：
- 信息组零差别（floor 0.005 < 实测 m_g 0.034~0.042 从未生效；
  混合版 c 比 1.01）→ 保护 0.13/0.01 不变；
- 退化组代价已量化：假开门率 23.8% vs 混合 17.8%，差 6pt；可接受因
  ① 除零不发生（无信息前缀扰动本底 m_g=0.0125，分母天然非零）
  ② 假开伤害有界（摊平无方向，仅损失该位锐化）③ 退化组罕见
  （teacher=成功 sibling 终端购买总带信息）；
- 纯形式独有性质：**尺度不变性**（TV 同乘常数 λ 不变，门只读本组
  散度地形的形状）——混合版的全局锚点破坏此性质。

**v5 冻结形式（用户拍板版）**：ω_v ∝ p_v^{1−λ}，
λ = TV/(TV + mean_group(TV))，仅成功侧，teacher=组内成功 sibling
短 outcome 前缀。一个统计量、零常数、尺度不变。

## 工程决策：v5 训练接入借鉴 SDAR/RLSD 框架，不借鉴其 skills 特权（2026-07-13）

审查了本仓（SDAR）已有的 OPSD 训练框架（`main_sdar`/`main_rlsd` →
`rlsd_ray_trainer.py`、`sdar_utils.py`、dp_actor 的 `use_sdar_loss` 开关）：

**借鉴的框架件**（照抄模式到 BEACON）：
1. `build_teacher_batch`：驱动侧把特权前置 prompt、左截断左 pad、拼回原
   response，构造标准键名的 teacher DataProto——FSDP+rmpad 下对齐无坑，已验证；
2. fit 挂接点：`old_log_prob` 之后、`compute_advantage` 之前插 teacher 前向，
   结果以 `batch.batch[...]` 传下去；
3. dp_actor 接入模式：`select_keys` 追加预计算张量 → 配置开关 →
   `policy_loss += extra * coef` → metrics；
4. 小件：`[Privileged ...]` 前缀格式；`_get_rlsd_lambda(global_steps)`
   调度器思路（v5 修正如早期不稳可加 warmup，先留接口）。

**不能复用的**：`compute_log_prob` 只回采样 token 的 logp，v5 的 TV 需要
全词表分布 → 仍需新增 `compute_v5_tv` worker 方法（骨架仿 compute_log_prob）。

**hindsight 不借鉴 skills，三条理由**：
1. 语义错位：skills 是事前通用方法论，非 outcome-conditioned；v5 要求
   teacher 按本任务实例的未来价值排序候选，只有事后信息能给；
2. 门污染：`ws_gate_null.py` 已证无信息前缀也抬背景 TV（meanTV=0.0125）,
   skills 长且与具体决策无关，TV 尖峰会落在风格 token 而非成败分叉；
3. 充分性阶梯：`ds_sib_teacher.py` 已证特权越偏离最小充分统计量越差
   （完整轨迹在深搜为负收益），skills 比完整轨迹更远。
   维持：WebShop=终端解，深搜=gold answer。

## 工程决策：v5 训练接入借鉴 SDAR/RLSD 框架，不借鉴其 skills 特权（2026-07-13）

审查了本仓（SDAR）已有的 OPSD 训练框架（`main_sdar`/`main_rlsd` →
`rlsd_ray_trainer.py`、`sdar_utils.py`、dp_actor 的 `use_sdar_loss` 开关）：

**借鉴的框架件**（照抄模式到 BEACON）：
1. `build_teacher_batch`：驱动侧把特权前置 prompt、左截断左 pad、拼回原
   response，构造标准键名的 teacher DataProto——FSDP+rmpad 下对齐无坑，已验证；
2. fit 挂接点：`old_log_prob` 之后、`compute_advantage` 之前插 teacher 前向，
   结果以 `batch.batch[...]` 传下去；
3. dp_actor 接入模式：`select_keys` 追加预计算张量 → 配置开关 →
   `policy_loss += extra * coef` → metrics；
4. 小件：`[Privileged ...]` 前缀格式；`_get_rlsd_lambda(global_steps)`
   调度器思路（v5 修正如早期不稳可加 warmup，先留接口）。

**不能复用的**：`compute_log_prob` 只回采样 token 的 logp，v5 的 TV 需要
全词表分布 → 仍需新增 `compute_v5_tv` worker 方法（骨架仿 compute_log_prob）。

**hindsight 不借鉴 skills，三条理由**：
1. 语义错位：skills 是事前通用方法论，非 outcome-conditioned；v5 要求
   teacher 按本任务实例的未来价值排序候选，只有事后信息能给；
2. 门污染：`ws_gate_null.py` 已证无信息前缀也抬背景 TV（meanTV=0.0125），
   skills 长且与具体决策无关，TV 尖峰会落在风格 token 而非成败分叉；
3. 充分性阶梯：`ds_sib_teacher.py` 已证特权越偏离最小充分统计量越差
   （完整轨迹在深搜为负收益），skills 比完整轨迹更远。
   维持：WebShop=终端解，深搜=gold answer。

## hindsight 来源定稿：own outcome（HER 式），废除组内 argmax（2026-07-13，`ws_hindsight_source.py`）

用户质疑"最优 sibling 终端解"的 argmax 步骤不优雅：正样本用自己的答案不就行了？
221 个残杀对上对比三种 teacher（门 λ=TV/(TV+mean_group(TV))，GRPO 基线 strip=0.602）：

| teacher | strip | siteLam | openBg | contrast |
|---|---|---|---|---|
| own（自己的终端购买） | 0.131 | 0.662 | 0.245 | **3.46** |
| best sibling（现方案） | 0.127 | 0.664 | 0.245 | 3.42 |
| gold（goal asin+options） | 0.154 | 0.625 | 0.219 | 3.26 |

- own 与 best **统计无差别**，且 own==best 的终端解重合率仅 5/10——一半情况特权
  内容真的不同，效果仍一致 → "最优"是多余计算；
- 担心的自我确认效应未发生：own 背景 meanTV=0.0382 ≈ best 0.0386，
  对比度不塌（门的尺度不变性 + 短 outcome 特权只在决策位点起作用）；
- gold 略差，因 goal 属性大半已在 instruction 里（agent 本来可见），
  teacher 增量被稀释；gold 的唯一隐藏增量（ASIN）own 也有；
- **通用形式统一**：teacher = 该轨迹自己的最终 outcome（HER 思想）。
  WebShop=own 最终购买（item+options），深搜=own 最终答案
  （成功侧与 gold 天然重合，EM=1 时同一字符串）。
  零 argmax、零跨轨迹计算、零 oracle 依赖，单成功组可用。

**v5 hindsight 冻结**：仅成功侧（组内 adv>0），特权=own outcome 短前缀。

## 通用形式复核：新鲜 rollout 双侧验证 + "奖励凭证"精化（2026-07-13 晚）

**WebShop 侧（`ws_ckpt_rollouts4.json`，全新任务 661~710，out-of-sample）**：
83 对，GRPO strip=0.462 → own 0.138 / best 0.139 / gold 0.139，
own==best 终端解重合仅 2/5。own 与 best 在没见过的任务上依旧无差别，
结论不是对旧 4 组的过拟合。

**深搜侧（`ds_own_teacher.py`，2wiki R=1 轨迹 21 条，126 个 fork）**：
- 天真形式翻车：raw prediction 只有 5% 与 gold 完全同串，多为整句复述
  （"The director ... was born in New York City..."）。整句作特权时
  spearman(q,V)=+0.14（gold +0.22），agree 位污染 0.156（gold 0.027）——
  多余词汇稀释并污染 teacher；
- **提取修复**：取 own prediction 中被奖励函数 EM 命中的那个 span 作特权
  → spearman +0.20（gold +0.22，配对 31% 胜 +23% 平），agree 污染 0.034，
  自门控 9.6×（gold 15.5×）。基本追平 gold；
- 注意：EM 命中的 span 本身就是 gold 字符串 → 深搜成功侧
  own-extracted 与 gold 内容重合，仅措辞框架不同。

**通用形式最终精化（比"own outcome"再进一步）**：
> 特权 = 奖励函数从该轨迹自身 outcome 中提取的**认证凭证**（reward certificate）。

- WebShop：凭证 = 最终购买（item+options）——奖励由此计算；
- 深搜：凭证 = 答案中被 EM/F1 命中的 span——奖励由此计算；
- 性质：teacher 看到的信息恰好等于奖励函数看到的信息，不多不少
  （最小充分统计量的构造性定义）；零 oracle、零 argmax、每轨迹自含；
- 教训：不能直接塞 raw 输出，"凭证提取"这一步就是奖励函数已做的操作，
  不引入新机制。

**实现层固定（用户确认）**：特权构造直接复用奖励函数的提取代码路径，
不写第二套：
- 深搜：`verl/utils/reward_score/search_r1_like_qa_em.py` 的
  `extract_solution` + `normalize_answer`/`em_check`，凭证 = 命中的
  gold alias（own-extracted 实验即此路径，+0.20 追平 gold +0.22）；
- WebShop：奖励在 env 内部算，无提取函数可 import；凭证 = 奖励函数的
  轨迹侧实参（最终购买 asin+options），从已执行的 click 动作重构，
  与 env 收到的输入逐字节一致；
- 部分得分（0.8 分/F1<1）的严格凭证是"匹配见证"，需 env 内部信息；
  实验证明购买本身够用（own 打平 gold），保持简单；
- 原则："teacher 信息 = 奖励信息"由代码复用保证，不靠约定。

## v5 训练接入完成 + 冒烟通过（2026-07-13 深夜，BEACON 仓）

实现件：`verl/trainer/ppo/v5_ccr.py`（凭证特权构造/teacher 序列/自校准门/
守恒修正）、`dp_actor.py`（`compute_v5_tv` 双前向 TV + rmpad/非 rmpad 两路
修正项）、`fsdp_workers.py` 注册、`ray_trainer.py` 两处挂接（特权构造在
adjust_batch 前保步序；TV+λ 在 old_log_prob 后）、`run_webshop_v5.sh`
（除 v5 开关外与 run_webshop.sh 逐行一致）。

单元测试（`tests_v5/test_v5_unit.py`）：λ=0 修正梯度==0（数值验证恒等式）；
λ=0.9 时 runner-up 梯度严格小于 GRPO（保护生效）；teacher 拼装对齐正确。

冒烟（2 卡、1 步）三情形：
1. 基座模型零成功率 → 正确退化（corr=0，指标与基线无异）；
2. step150 ckpt 成功率 0.5 → 暴露 bug：全成功组被 `>group_mean` 排除
   （teacher 全空）；而全成功组正是步级残杀最强处；
3. 修复为"≥组均值且>0"后：λ_mean=0.113、open_frac=0.123（与探针 openBg
   量级一致）、v5_corr=-0.263 非零、grad_norm 4.2 正常、跑通无异常。

开销：TV 前向 12.4s/步（小 batch，未走 rmpad，双前向）；正式训练估计
每步 +1~2 分钟，可后续优化为 rmpad 路径。

**rmpad 优化落地（同晚）**：照 SDAR teacher 前向的 rmpad 模式重写
`compute_v5_tv`，并加一步裁剪——用 response 掩码从 rmpad 行中只选出
"预测 response token 的行"再做全词表 softmax/TV（prompt/padding 位置
不算概率）。实测 12.4s → 3.2s（3.9×），与 old_log_prob（2.5s）同量级；
门指标与优化前一致（λ_mean 0.112、open_frac 0.123、corr -0.259），
grad_norm 正常。非 rmpad 配置保留 padded 兜底路径。

**成功侧判据定稿**：ep_reward ≥ mean_group(ep_reward) 且 ep_reward > 0
（≥ 保留全成功组——episode 优势为 0 但 GiGPO 步级优势非零，残杀主战场）。

## v5 全量训练崩溃验尸（2026-07-15，run gzlpxf1v，gigpo_v5ccr_qwen2.5_1.5b）

**结果**：val task_score 全程 ~0.2（基线 0.8+），93 步后放弃。死亡时间线
（wandb output.log 逐步指标）：

- step 1–10 急剧锐化：熵 1.10→0.06（基线同期 0.79），ppo_kl 为基线
  10–100 倍，KL(ref) 7 步到 0.66（基线 0.02）；奖励反而涨得比基线快
  （step9 1.43 vs 0.82）——过度锐化的收割假象；
- step 11–12 崩溃：响应长度 100→512 顶格，奖励归零（重复输出退化吸引子）；
- step 12+ 脑死亡：全失败→无成功轨迹→teacher 建不出→corr≡0→
  grad_norm 0.02，死策略空转 80 步。

矛盾即线索：λ_mean 仅 0.01、open_frac ~1%，v5_corr 值却 -0.5~-1.7
（与 pg_loss 同量级）。

**根因：逐位置 logit 空间的守恒是虚构，参数空间不守恒。**

1. z_v = W_v·h，词表矩阵全位置共享。GRPO 反质量 ∝p_v，15 万尾部 token
   梯度≈0、embedding 从不被碰；v5 摊平后**每个门控位置给全词表一个方向
   一致的下压力**（仅成功侧→优势恒正→力永不反向），经共享 W_v/h 相干
   叠加：尾部 logit 在所有位置（不只门控位）被系统压低 = 全模型温度锐化
   = 全局熵崩塌。门开 1% 却 10 步崩干全局熵，由此解释。
2. 逐位置"保护"经不起参数投影：参数梯度是全位置 rank-1 项之和，位置内
   重分配是被淹没的微小分量。
3. 信任域不对称：pg 有 ratio clip，corr 无任何 clip/衰减，单向永动。

**证据早已在手，当时误读**（流程教训，权重排序错误）：7-13 真梯度 A/B
（ws_v5_trainstep.py）里 v5 的 H@sites 一个优化步 0.807→0.262（GRPO
→0.998 稳定），且 p(y_b) 保护只剩 2pp（0.290 vs 0.284，场模拟预测
0.60→0.13 的巨大效果消失）。当时只盯 p(y_b) 略优放行，把熵崩塌记为
"Adam 瞬态污染不作依据"。**场模拟系统性高估收益、完全看不见毒性；
真梯度实验的熵信号才是一票否决项。**

**裁决：目标保留，杠杆废除。**

- 资产保留：残杀病理为真（succ-succ 60% 实测）；own-outcome 奖励凭证
  teacher + 自校准 TV 门是便宜精准的决策位探测器——继续用；
- 废除："附加损失改写全词表反质量分配"整条杠杆。任何绕开 clip 框架、
  直接向词表空间注入梯度的方案都会撞上同一堵参数空间泄漏的墙；
- 安全通道唯一：**标量优势通道**（PPO 全部安全机制所在）。候选：
  - v6a 门控优势衰减：成功侧 A_i ← A_i·(1−κλ_i)。不注入新梯度，只在
    残杀位少下毒；结构上只会减小更新幅度，不可能熵崩塌。复用全部 v5
    基础设施，仅把 corr 损失换成优势衰减（~5 行）；
  - v6b 已验证兄弟共享信用：GiGPO 锚点分组内 ≥2 成功轨迹分叉处，给
    兄弟动作 token 一份小正优势（有限个已验证 id，非全词表），走标准
    clipped pg（token 级 RLOO 式信用共享）；
  - 旁证反转：TECA 的加性优势提升当时被评"不如 v5 优雅"，但它待在
    标量优势通道里——保守是对的。

下一步：先 v6a（本次诊断的直接对偶：v5 是"往别处推"，v6a 是"别推
那么狠"），GiGPO vs GiGPO+v6a 各 150 步，盯熵衰减速率/死组比例/成功率。

## v6 门定稿：凭证交换门，κ 消元（2026-07-15，`ws_cert_swap_gate.py`）

用户质询 A_i(1−κλ_i) 中 κ 的来历。诊断：**κ 是给错配的门打的补丁**。
旧门 λ=TV(p,q_own) 混合两种相反情形——(1) 任意承诺位（兄弟证明其他选
择同样成功，反事实优势=0，该全额衰减）；(2) 共识修正位（所有成功结局
要求同一 token，信用该保留甚至加强，TECA 方向）。学生-teacher TV 分不
开两者，只好用 κ<1 折衷——与当年 c=0.03 同病。

**修法：换门。凭证交换门 λ^frat = TV(q_own, q_sib) 自校准 odds 形式**，
q_sib = 同组另一条成功轨迹的凭证做前缀的 teacher。两份凭证分歧处=任意
承诺位（开门），一致处=共识位（关门）。κ 消失：门开即反事实优势为 0，
全额衰减是正确极限而非可调强度。k=1（组内无第二条成功轨迹）时 q_sib
不存在 → λ≡0 自动关门（无兄弟可残杀则机制自动失效，旧门无此性质）。
成本不变：TV 双前向从 (p, q_own) 换成 (q_own, q_sib)。

**out-of-sample 探针**（rollouts4，83 残杀对=61 解法叉+22 措辞叉，
3009 位置）：

| | lam_old | lam_new |
|---|---|---|
| 解法叉（买了不同商品） | 0.628 | 0.595 |
| 措辞叉（同一购买） | 0.587 | **0.000** |
| 背景 | 0.215 | 0.139 |
| openBg(>0.5) | 0.242 | **0.156** |

- 解法叉：新门照常开（衰减因子 0.40，剥夺削 ~60%）；
- 共识保留：旧门开的 729 个位置中 **45.4%** 新门关闭——这些是旧门会
  错杀的信用，正是 κ 想保护而保护不精准的部分；
- 措辞叉 λ_new≡0 是**构造性的**：同解法兄弟凭证相同 → q_own=q_sib。
  判读为特性而非缺陷：**门继承奖励函数的等价关系**——奖励分不出的差
  别（同一购买的不同措辞），门也不保护，允许措辞侧正常固化（熵预算
  花在解法多样性上）。解法多样性（pass@k/方差组存活的载体）恰是新门
  全保护的部分。措辞叉待遇=GiGPO 原样，不劣化。

**v6 冻结形式**：A_i ← A_i·(1−λ^frat_i)，λ^frat=TV(q_own,q_sib)/(·+m_g)，
仅成功侧，q_sib 取组内得分最高的其他成功轨迹凭证。人为预设参数 0 个
（κ 已消元）。纯优势缩放，在 clipped pg 框架内，结构上无熵崩塌通道。

## 战略重定位：GRPO+v6 vs GiGPO 的真实战场（2026-07-15）

用户重申目标：**GRPO+idea 打过 GiGPO**（不是 GiGPO+idea）。由此发现
一个此前未言明的等价关系和一组实测事实：

**GiGPO 的步级机制 = v6 的硬锚点特例。** GiGPO 步级优势 = 本步回报 −
同锚点（完全相同观测）各动作回报均值。当两条成功轨迹在同一锚点选不同
动作且回报相等时，步级优势 = 0——**GiGPO 早已在"它看得见的分叉"上做
优势清零**，这正是 v6 的衰减语义。v6 是同一原理的软化推广：把"完全相同
状态匹配"换成"凭证交换 TV"，把步级粒度换成 token 级，把"等回报才清零"
换成"兄弟凭证分歧即任意承诺"。

**锚点复现率实测**（rollouts4，8 组×4）：WebShop 步数 186，跨轨迹锚点
复现 99%、含动作分叉 75%——WebShop 是 GiGPO 主场，步级机制火力全开。
深搜索结构性相反：锚点 = 完整观测（含自己生成的 query + 检索结果），
step 1 之后兄弟间完全相同状态概率≈0，GiGPO 步级机制退化 → GiGPO≈GRPO
（ds_fork_anchor_rows 中各 rollout 的 sq 查询互不相同佐证）。凭证 teacher
不需要状态复现，深搜索照常工作。

**诚实的胜负评估**：
- 深搜索：非对称战场，GiGPO 缴械、v6 满血——**主攻点，胜面大**；
- WebShop：GiGPO 满血，且步级优势覆盖成败两侧；v6 只做成功侧、开门率
  15.6%，优势是 token 粒度（GiGPO 步内均匀）与无需等待状态匹配。
  GRPO vs GiGPO 基线曲线本就接近（val ~0.8 重叠），目标是持平或略胜；
- 已证：门定位、结构安全、解法叉保护；**未证：残杀消除→成功率提升的
  因果链**（GiGPO 带着残杀也能到 0.8，病理的性能代价还没被直接测过，
  可能兑现在 pass@k/后期平台期/深搜索而非 WebShop 峰值）；
- 若 WebShop 上纯衰减不够：v6b=守恒信用重塑（衰减掉的信用按轨迹守恒
  转移到共识修正位，token 级复刻 GiGPO 的"信用集中"，代价+1 次学生
  前向）。留作二级弹药，不进首发。

首发实验：GRPO / GRPO+v6 / GiGPO 三臂各 150 步 WebShop（盯成功率、
熵衰减、pass@k、死组比例）；深搜索 GRPO+v6 vs GiGPO 为决胜局。

## 修正：gold 唯一性决定武器-战场匹配（2026-07-15，用户质询）

用户问"gold answer 不是唯一的吗"，暴露上节"深搜索主攻点"论断的漏洞。

**WebShop gold 不唯一（实测）**：奖励=购买商品对指令规格的属性/选项/
价格匹配，不检查 ASIN 等同。rollouts4 铁证：task 2 两条成功轨迹买
b09p39qn2w（gold 是 b0969g2dh8）拿满分 1.00；task 0 三条全买非 gold
商品拿 0.8/0.9，且选项组合互不相同。解法叉在成功侧真实存在，
相残门有活干。

**深搜索 gold 近乎唯一 → v6 相残门自动全关**：成功兄弟凭证几乎总是
同一答案 span → q_own≈q_sib → λ_frat≡0 → GRPO+v6 退化为 GRPO。
退化安全（门继承奖励等价关系：路径差异奖励分不出，门也不保护），
但"深搜索 v6 胜面大"不成立——GiGPO 缴械+v6 缴械=三方平局。

**修正后的武器-战场匹配**：
- WebShop（结局层任意承诺）：武器=v6 相残衰减；
- 深搜索（路径层差异，奖励不可分）：武器=信用集中，用旧门 TV(p,q_own)
  + 已验证的方向项 log(q/p)（+0.217/60%，单 gold 信息上限）。零参数
  候选形式：轨迹内守恒归一 w_i=(1+λ_i)/mean(1+λ)（总信用不变，
  从安静位向凭证敏感位转移）。设计待推敲，为深搜索决胜局的独立组件。

## 环境约定

- **GPU 使用：只用卡 5（`CUDA_VISIBLE_DEVICES=5`），不要占用 0/1/2/3。**
- 深轨迹生成期间临时借用卡 6/7 跑 vLLM，生成结束即释放。
