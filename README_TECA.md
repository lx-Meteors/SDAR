# TECA: Teacher-Entropy Credit Assignment

基于教师-学生熵差的 token 级信用分配方法(本分支在 SDAR 官方代码基础上实现)。在 GRPO 序列级优势的基础上,利用带特权信息的 teacher 模型识别"学生过早坍缩、教师知道存在多个合理选项"的决策 token,只对这些位置加性地追加优势。

- 训练框架:SDAR(verl-agent fork),webshop 多轮 agent 任务
- 除优势塑形外,rollout / 蒸馏 loss / PPO 更新与 SDAR 基线完全一致,便于隔离增量

---

## 快速开始

环境安装与 SDAR 官方 README 完全一致(conda `sdar` + `verl-webshop` 双环境),无额外依赖。

```bash
# TECA 训练(3B, 4 GPU, webshop)
export WANDB_API_KEY=your_key_here
bash examples/teca_trainer/run_webshop_3b_4gpu.sh

# 严格对齐的基线(除方法外配置完全相同)
bash examples/sdar_trainer/run_webshop_3b_4gpu.sh    # SDAR
bash examples/grpo_trainer/run_webshop_3b_4gpu.sh    # GRPO
bash examples/gigpo_trainer/run_webshop_3b_4gpu.sh   # GiGPO

# 单元测试(CPU 闭式熵一致性 + GPU dp_actor 前向统计正确性)
python tmp_teca/test_teca_unit.py
```

## 代码结构

| 文件 | 作用 |
|---|---|
| `verl/trainer/main_teca.py` | 训练入口(镜像 `main_sdar.py`) |
| `verl/trainer/ppo/teca_ray_trainer.py` | TECA trainer:teacher 前向、δH 计算、优势塑形的编排 |
| `verl/trainer/ppo/teca_utils.py` | 核心算子:去目标熵闭式解、资格条件、塑形公式 |
| `verl/workers/actor/dp_actor.py` | 前向统计:old_log_prob 同一次前向顺带输出全词表熵/argmax/realized log-prob(`calculate_teca_stats`) |
| `verl/workers/fsdp_workers.py` | worker 侧透传 TECA 统计量 |
| `examples/teca_trainer/run_webshop_3b_4gpu.sh` | 启动脚本(超参见 §8) |
| `tmp_teca/` | 离线验证脚本与单测(信号检查、选择器对比、三方信用分配对比) |

---

## 1. 问题

GRPO 的优势是序列级的:一条轨迹内所有 token 共享同一个标量优势

$$
A_i = \frac{R_i - \mathrm{mean}(R_1,\dots,R_G)}{\mathrm{std}(R_1,\dots,R_G)}
$$

其中 $R_i$ 是第 $i$ 条 rollout 的回报,$G$ 是组大小。这意味着成功轨迹里的每个 token——无论是真正的关键决策还是随手写的虚词——拿到完全相同的信用。轨迹越长,这种均摊越稀释关键步骤的学习信号(长程信用分配问题)。

TECA 的目标:在**不引入额外监督**的前提下,把成功轨迹的信用向少数关键决策 token 集中。

## 2. Teacher 的构造

复用 SDAR 的 skill-augmented teacher:同一份模型权重,但 prompt 前拼接特权 skill 文本(任务相关的策略提示,训练后不可用):

$$
\pi_S(\cdot \mid x, y_{<t}) = \pi_\theta(\cdot \mid x, y_{<t}), \qquad
\pi_T(\cdot \mid x, y_{<t}) = \pi_\theta(\cdot \mid \underbrace{\text{skill}}_{\text{特权信息}},\, x,\, y_{<t})
$$

teacher 和 student 的差异**只来自条件中的特权信息**,不来自权重差异。因此两者分布的差异可以解释为"特权知识对该位置的影响",而不是模型能力差异。

## 3. 核心信号:去目标熵差 $\delta H$

### 3.1 去目标(target-excluded)分布

在响应的第 $t$ 个位置,实际采样到的 token 为 $y_t$。把 $y_t$ 从词表中剔除,对剩余候选重新归一化:

$$
q_v = \frac{p_v}{1 - p_{y_t}}, \qquad v \in \mathcal{V} \setminus \{y_t\}
$$

**含义**:我们要度量的不是"这个位置整体多不确定",而是"**除了已经选中的那个词,备选方案的分布有多宽**"。把 $y_t$ 本身排除,是因为它的概率高低已经由 log-prob 项(PPO ratio)刻画,混进熵里只会稀释"备选宽度"这个语义。

### 3.2 去目标熵及其闭式解

去目标熵定义为

$$
H^{\setminus y}
= -\sum_{v \neq y_t} q_v \log q_v
$$

直接计算需要整个词表的概率。但它可以由**全词表熵 $H_{\text{full}}$ 和目标概率 $p_y$ 两个标量**闭式恢复。推导:记 $\sum_v p_v \log p_v = -H_{\text{full}}$,则

$$
\begin{aligned}
H^{\setminus y}
&= -\frac{1}{1-p_y}\sum_{v \neq y}p_v\bigl(\log p_v - \log(1-p_y)\bigr) \\
&= -\frac{1}{1-p_y}\Bigl(-H_{\text{full}} - p_y \log p_y\Bigr) + \log(1-p_y) \\
&= \frac{H_{\text{full}} + p_y \log p_y}{1-p_y} + \log(1-p_y)
\end{aligned}
$$

**工程意义**:训练时每个位置只需存 $H_{\text{full}}$ 和 $\log p_y$ 两个标量(而非 top-k 候选或整个词表),显存与通信开销可忽略。实现见 `teca_utils.entropy_excluding_target_closed_form`(单测中闭式与掩码直算的最大误差 2e-6)。

### 3.3 熵差

$$
\delta H_t = H_T^{\setminus y}(t) - H_S^{\setminus y}(t)
$$

**含义**:去掉已选 token 后,teacher 的备选分布比 student 宽多少。

| 位置类型 | $\delta H_t$ | 解释 |
|---|---|---|
| 两者都不确定 | $\approx 0$ | 本质困难位置,特权信息也无法消解,不该给额外信用 |
| student 确定、teacher 更发散 | $> 0$ | **student 过早坍缩**:teacher 凭特权知识知道此处存在多个合理选项,是真正的决策分叉点 |
| student 纠结、teacher 确定 | $< 0$ | teacher 用特权知识消除了纠结(这是 SDAR 蒸馏 loss 已经处理的方向) |

关键性质:$\delta H$ 是**对比信号**。单侧的 student 熵会把"两者都不确定"的噪声位置也选进来(第一行),而 $\delta H$ 通过 teacher 基准把它对消掉。离线验证(§7)证实两者选出的 token 几乎不重合。

## 4. 资格条件(哪些 token 被塑形)

四个条件同时满足才塑形,逐条含义:

**(a) 正优势样本**

$$
A_i > 0
$$

只对成功轨迹追加信用("只奖不罚")。失败轨迹的 token 保持 GRPO 原值,避免在错误行为上做 token 级判断——离线对比显示 GiGPO 恰恰在失败轨迹上系统性帮倒忙(§7)。

**(b) top-1 一致**

$$
y_t = \arg\max_v \pi_T(v) \quad \text{且} \quad y_t = \arg\max_v \pi_S(v)
$$

实际选中的 token 必须同时是 teacher 和 student 的第一选择。这保证我们奖励的是"两个分布都认可的正确决策",而非 student 侥幸采到的低概率 token;也保证 $\delta H$ 比较的是同一个"最优解之外的备选集"。

**(c) 熵差取正部**

$$
\delta H_t > 0
$$

只保留"teacher 备选更宽"的方向。负方向(teacher 更确定)已由 SDAR 蒸馏 loss 的门控前向 KL 处理,重复施加会双重计费。

**(d) 行内 top-20% 分位数筛选**

$$
\delta H_t \ \geq\ \mathrm{Quantile}_{1-\rho}\bigl(\{\delta H_{t'} : t' \text{ 满足 (a)(b)(c)}\}\bigr), \qquad \rho = 0.2
$$

在每条响应内部,只取已合格 token 中 $\delta H$ 最高的 20%。**动机来自实证而非先验**:扩样调查发现仅用 (a)(b)(c) 时约 40% 的 token 被加权、其中大量是虚词,信号被稀释;按行内分位数强筛后,被加权比例降到约 14%,实义词占比上升。分位数按行计算,使每条轨迹被集中的 token 数与其长度成比例,不受跨样本长度差异干扰。

## 5. 优势塑形

对满足全部资格条件的 token:

$$
A'_{i,t} = A_{i,t} + \beta \cdot \delta H_t, \qquad \beta = 0.1
$$

其余 token 的优势保持不变。

**设计选择**:
- **加性而非乘性**:$A(1+\beta\delta H)$ 会把塑形幅度与优势大小耦合(优势大的轨迹被塑形得更多),加性保证同一 $\delta H$ 在不同轨迹上贡献相同的绝对增量。
- **无门控阈值、无截断**:$\beta=0.1$ 由 β 扫描确定——在真实 rollout 上该幅度下**零符号翻转**($A'$ 不会把正优势变负、负变正),扰动温和且可测,因此不需要额外的 clip 机制。
- $\delta H$ 的实测量级:webshop 上合格 token 的 $\delta H$ 均值约 0.3 nat,q99 约 0.5 nat,故 $\beta\delta H$ 的典型增量在 0.03–0.05,是 GRPO 优势(z 分数,量级 ~1)的温和修正。

## 6. 完整训练流程

每个训练步(与 SDAR 唯一的差异是第 4、6 步):

1. **Rollout**:vLLM 多轮采样,得到轨迹组与环境回报 $R_i$。
2. **old_log_prob 前向**(student):计算 $\log \pi_S(y_t)$;**同一次前向**顺带输出 student 的 $H_{\text{full}}^S$、$\arg\max$、$\log p^S_{y}$(`calculate_teca_stats` 开关,不增加前向次数)。
3. **teacher 前向**:对拼接 skill 的 teacher batch 做一次 no-grad 前向,输出 $H_{\text{full}}^T$、$\arg\max$、$\log p^T_{y}$;其中 $\log p^T_{y}$ 同时复用为 SDAR 蒸馏 loss 的 teacher log-prob(**不额外增加前向**)。
4. **计算 $\delta H$**:用 §3.2 闭式解分别得 $H_T^{\setminus y}, H_S^{\setminus y}$,相减。
5. **GRPO 优势**:组内 z 分数,广播到 token。
6. **TECA 塑形**:按 §4 资格条件 + §5 公式修改 `advantages`。
7. **损失与更新**:PPO clip loss(用塑形后的优势)+ SDAR 门控前向 KL 蒸馏 loss + KL 正则,全部与 SDAR 基线一致。

单步开销:相对 SDAR 只多了 teacher 前向里的全词表统计(fp32 分块、单遍扫描,max/argmax 与 logsumexp 融合)。实测 3B/4×A800:TECA 约 365 s/步 vs SDAR 约 340 s/步。

## 7. 实证依据(设计决策的来源)

以下均在真实 webshop rollout(真实环境回报)上离线验证,脚本见 `tmp_teca/`:

**(i) $\delta H$ 与"直接用熵筛选"是不同的信号,且更对。** 七种选择器各取行内 top-20%,对比选中 token 命中 instruction 约束词的比例与落在 action 决策段的比例(两批独立数据,seed 7/11,结论一致;脚本 `offline_selector_compare.py`):

| 选择器 | 命中约束词 | 落在 action 段 | 与 $\delta H$ 选集 Jaccard |
|---|---|---|---|
| $\delta H > 0$(TECA) | 0.201 | **0.145** | 1.00 |
| student 全词表熵 | 0.150 | 0.013 | 0.12 |
| student 去目标熵 | 0.152 | 0.068 | 0.09 |

student 自身熵挑出的是"模型措辞纠结"的位置(选中 token 平均熵 1.40,为全局均值 3 倍),几乎从不落在 action 决策段(1.3%);$\delta H$ 挑出的位置 student 自己并不纠结(熵≈均值 0.42),是 teacher 凭特权信息才识别出的分叉点。两个选集重合度只有 ~0.1:**teacher 前向买到的是 student 单侧拿不到的正交信息**。

**(ii) 信用分配质量优于 GRPO 与 GiGPO。** 96 条真实轨迹、oracle step 标签(点金标/点错/搜索是否召回金标),轨迹内配对 AUC(好 step 的信用是否排在坏 step 之上;脚本 `offline_three_way_credit.py`):

| 方法 | 全部轨迹 | 仅正优势轨迹 |
|---|---|---|
| GRPO / SDAR | 0.500 | 0.500 |
| GiGPO | 0.427 | 0.806 |
| TECA | **0.561** | 0.719 |

配对 bootstrap:TECA−GRPO = +0.061(95% CI [0.010, 0.115]),TECA−GiGPO = +0.134(CI [0.044, 0.217])。GiGPO 在正轨迹上更强,但在失败轨迹上系统性反向(好的搜索步骤因与成功轨迹共享锚点而拿到负信用),且锚点存活率仅 59%;TECA 不塑形负样本,失败轨迹保持 0.5,从不帮倒忙。两者信号来源正交,原则上可叠加。

**(iii) 各资格条件的消融依据。** 纯 relu(无 (d)):40% token 被加权、虚词占比高;行内中心化:改善有限;top-20% 分位数:加权比例 ~14%,实义词占比最高——故固化 (d)(脚本 `offline_effect_survey.py`)。

## 8. 超参数

| 参数 | 值 | 确定方式 |
|---|---|---|
| `beta` | 0.1 | β 扫描:≤0.2 时零符号翻转,取有可测信号的温和值 |
| `top_frac` | 0.2 | 效果调查:40%→14% 加权比例,实义词占比最优 |
| `positive_only` | true | 只奖不罚;避免在失败轨迹上做 token 级判断 |
| `require_top1` | true | 保证奖励的是双方认可的决策,且备选集可比 |
| `positive_dh_only` | true | 负方向已由 SDAR 蒸馏处理 |
| SDAR 蒸馏(`sdar_coef=0.01, gate_beta=5.0`) | 不变 | 与 SDAR 基线严格对齐,隔离塑形的增量 |

## 9. 与相关方法的关系

- **SDAR**:蒸馏 loss 把 student 拉向 teacher 的实现 token 分布(处理 $\delta H<0$ 方向);TECA 在优势侧处理 $\delta H>0$ 方向(student 过早坍缩),二者互补,TECA 训练时保留 SDAR loss。
- **GiGPO**:从环境侧(相同观测锚点分组)做 step 级信用;TECA 从模型侧(分布对比)做 token 级信用。GiGPO 依赖"不同 rollout 撞见相同观测"(webshop 上仅 59% 步骤存活)且在失败轨迹上有系统性反向;TECA 无此依赖。信号正交,可叠加。
- **熵正则 / 高熵 token 加权**:单侧 student 熵度量的是"模型在纠结",与"任务关键"在 agent 任务上几乎无关(§7-i);TECA 的对比熵差才把"特权知识改变确定性"的位置分离出来。
