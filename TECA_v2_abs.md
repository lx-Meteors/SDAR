# TECA v2_abs:双侧、恒正的教师-学生熵差信用分配(研究方案)

在生产版 TECA(单侧 `pos`)基础上做的一个**两侧对称化**变体。核心改动一句话:**不再只奖励"teacher 备选更宽"($\delta H>0$)的位置,而是把"teacher/student 去目标候选熵差够大"的位置——无论哪个方向——都视为信息量大的决策位,统一给正信用。**

- 训练框架:SDAR(verl-agent fork),webshop 多轮 agent 任务(与 TECA-pos 完全一致)
- 代码:`verl/trainer/ppo/teca_utils.py`(`compute_teca_advantage(..., variant="abs")`)、`verl/trainer/ppo/teca_ray_trainer.py`(读取 `algorithm.teca.variant`)
- 启动:`examples/teca_trainer/run_webshop_3b_4gpu_v2_abs.sh`
- 开关:`+algorithm.teca.variant=abs`(默认 `pos`,完全等价现有生产行为)

> 前置阅读:`TECA.md`(问题定义、teacher 构造、$\delta H$ 闭式解、资格条件、优势塑形与实证依据)。本文只描述 v2_abs 相对 pos 的**增量**,不重复共有部分。

---

## 1. 动机:为什么要考虑 $\delta H<0$ 的一侧

生产版 TECA 的资格条件 (c) 只保留 $\delta H_t>0$,理由是"负方向($\delta H<0$,student 比 teacher 在备选集上更纠结)已由 SDAR 蒸馏 loss 的门控前向 KL 处理"(见 `TECA.md` §4c、§9)。

v2_abs 提出一个对称假设:

> **teacher 与 student 在去目标候选集上的熵差幅度 $|\delta H_t|$,本身就是"该位置是否为决策分叉点"的度量,与差的符号无关。**

- $\delta H>0$:student 过早坍缩、teacher 知道有多个合理备选(pos 已覆盖)。
- $\delta H<0$:student 在备选上发散、teacher 凭特权信息把备选收窄——**这同样是一个"特权知识显著改变了备选结构"的位置**,只不过方向相反。v2_abs 认为这里也应当把信用推向实际选中的 token $y_t$(它是双方 top-1)。

因此 v2_abs 用 $|\delta H|$ 作为选择键与信用幅度,并对两侧都给**正**信用(纯加性、只奖不罚,与 pos 一致)。

## 2. 与 TECA-pos 的差异(仅三处)

| | TECA-pos(生产) | **TECA v2_abs** |
|---|---|---|
| 资格 (c) 方向 | `top1 & (δH > 0)` | `top1 & (δH ≠ 0)` |
| 行内 top-20% 选择键 | 按 `δH` 取分位 | 按 `\|δH\|` 取分位 |
| 信用项 | `β · relu(δH)` | `β · \|δH\|`(两侧都正) |

资格条件 (a) 正优势样本、(b) top-1 一致、(d) 行内 top-20% 分位数强筛,以及 $\beta=0.1$、加性、无截断、SDAR 蒸馏 loss 保持不变——全部与 pos 严格一致,保证是 apples-to-apples 的单一变量对照。

## 3. 优势塑形公式

对满足资格的 token:

$$
A'_{i,t} = A_{i,t} + \beta\,\lvert\delta H_t\rvert, \qquad \beta = 0.1
$$

资格指示(逐 token):

$$
\text{keep}_t=\underbrace{\mathbb{1}[A_i>0]}_{\text{(a) 正优势}}\cdot
\underbrace{\mathbb{1}[y_t=\arg\max\pi_T=\arg\max\pi_S]}_{\text{(b) top-1 一致}}\cdot
\underbrace{\mathbb{1}[\delta H_t\neq 0]}_{\text{(c) 双侧}}\cdot
\underbrace{\mathbb{1}[\lvert\delta H_t\rvert\ge\tau^{\text{row}}_{1-\rho}]}_{\text{(d) 行内 }|\delta H|\text{ 前 }20\%}
$$

行内幅度阈值:

$$
\tau^{\text{row}}_{1-\rho}=\mathrm{Quantile}_{1-\rho}\!\left(\{\lvert\delta H_{t'}\rvert:t'\text{ 满足 (a)(b)(c)}\}\right),\qquad\rho=0.2
$$

其余 token 优势不变。对照:pos 为 $A'=A+\beta\,\mathrm{relu}(\delta H)$、按 $\delta H$ 选分位。

## 4. 研究假设

- **H1(覆盖率)**:纳入 $\delta H<0$ 一侧后,被塑形的决策 token 覆盖增加,合格 token 中实义/动作词占比不下降。
- **H2(信用质量)**:v2_abs 的轨迹内配对 AUC(`TECA.md` §7 口径)$\ge$ pos,尤其在"student 在选项上纠结、teacher 果断"的位置能补上 pos 拿不到的信用。
- **H3(训练效果)**:webshop 成功率/回报的 AUC 与收敛速度 $\ge$ pos 且 $\ge$ SDAR 基线。

## 5. 实验方案

**对照组(全部 3B / 4×GPU,除 variant 外超参逐字对齐):**

| 组 | 启动 | 关键差异 |
|---|---|---|
| SDAR 基线 | `examples/sdar_trainer/run_webshop_3b_4gpu.sh` | 无优势塑形 |
| GRPO 基线 | `examples/grpo_trainer/run_webshop_3b_4gpu.sh` | 无 teacher |
| TECA-pos | `examples/teca_trainer/run_webshop_3b_4gpu.sh` | `variant=pos` |
| **TECA v2_abs** | `examples/teca_trainer/run_webshop_3b_4gpu_v2_abs.sh` | `variant=abs` |

**主指标**:webshop val 成功率 / 平均回报的训练曲线与 AUC(相同 step 预算)。
**过程指标(已在 `compute_teca_advantage` metrics 中)**:`teca/eligible_token_frac`、`teca/delta_h_eligible_mean`、`teca/dh_eff_eligible_mean`、`teca/adv_abs_change_mean`、`teca/top1_agree_frac`;与 pos 同盘观察合格 token 比例与优势扰动幅度是否落在温和区间。
**符号翻转监控**:统计 $\mathrm{sign}(A')\neq\mathrm{sign}(A)$ 的 token 比例(v2_abs 恒正、只作用于正优势样本,理论上不应翻转;需实测确认量级)。

**消融**:
1. `variant∈{pos, abs, neg}`(`neg` 单侧只取 $\delta H<0$,隔离"负方向本身是否有用")。
2. `top_frac∈{0.1, 0.2, 0.4}`:检验 v2_abs 对选择性强弱的敏感度。
3. 去 `require_top1`(`+algorithm.teca.require_top1=false`):检验放开 top-1 门控对决策 token 覆盖的影响。

**采纳准则**:以相同 step 预算下 val AUC 与过程指标为准,若 v2_abs 相对 pos 有稳定优势则采纳为默认,否则保留 pos。

## 6. 运行方式

```bash
cd /home/test/yyy/SDAR
# 默认 variant=abs, beta=0.1, top_frac=0.2
bash examples/teca_trainer/run_webshop_3b_4gpu_v2_abs.sh

# 覆盖超参(环境变量)
TECA_BETA=0.1 TECA_TOP_FRAC=0.1 bash examples/teca_trainer/run_webshop_3b_4gpu_v2_abs.sh
# 消融 neg / 放开 top1(追加 hydra 覆盖)
bash examples/teca_trainer/run_webshop_3b_4gpu_v2_abs.sh vllm +algorithm.teca.variant=neg
bash examples/teca_trainer/run_webshop_3b_4gpu_v2_abs.sh vllm +algorithm.teca.require_top1=false
```

`experiment_name` 形如 `teca_v2abs_qwen2.5_3b_beta0.1_top0.2_4gpu`,wandb project `verl_agent_webshopv1`,与 pos 同盘对比。

## 7. 风险与预期

- **选择键敏感性**:$|\delta H|$ 的分位选择在不同 token 类型上的分布需要监控;可通过 `top_frac` 与 `require_top1` 消融观察合格 token 的构成。
- **双重计费**:$\delta H<0$ 方向同时被 SDAR 蒸馏 loss 与 v2_abs 优势侧作用,可能对同一位置双重推动。需观察训练是否比 pos 更"急"(KL、熵、成功率震荡);必要时下调 `sdar_coef` 做交叉消融(列为二级实验)。
- **恒正、只作用正优势**:理论上不引入负向错配,符号翻转应为 0;这是相对有符号变体(`β·δH`)的安全性优势。

## 8. 代码改动点(已落地)

1. `verl/trainer/ppo/teca_utils.py::compute_teca_advantage`:新增 `variant: str = "pos"` 形参;`abs` 分支用 `δH≠0` 资格、`|δH|` 作行内分位选择键与信用幅度;`neg` 分支对称;`pos` 分支保持原逻辑(含 `positive_dh_only`),**默认行为零变化**。
2. `verl/trainer/ppo/teca_ray_trainer.py`:`__init__` 读取 `self.teca_variant = teca_cfg.get("variant", "pos")`,并传入 `compute_teca_advantage(..., variant=self.teca_variant)`。
3. `examples/teca_trainer/run_webshop_3b_4gpu_v2_abs.sh`:镜像 pos 启动脚本,仅多 `+algorithm.teca.variant=abs` 与新的 `experiment_name`。
