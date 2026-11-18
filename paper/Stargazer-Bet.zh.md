# Stargazer：用递归状态控制因果读取

**Bet 双层原型、Chain5 消融与复现协议**  
研究稿 · 2026 年 11 月 18 日 · 作者栏待定

## 摘要

Stargazer 是一种因果序列块。它用并行微型投影生成当前词元的局部统计，再用递归更新的 32 维状态 $Z_t$ 构造查询，从可见前缀读取键和值。这里把 $Z_t$ 称为全局结论瓶颈。它限制查询状态的维数，块内仍保留前缀键值和残差。冻结已有 Aleph 底座后，只读取第一层末词元的线性或门控 MLP 约有 6% 的 Chain5 正确率。恢复已训练的末端注意力后，同一个双层门控结构达到 79.05%。在相近参数量和相同训练预算下，重新拟合的第二个递归 Stargazer 达到 43.31%，重训的 Transformer 末端达到 20.51%。两层继续联合训练后，175,635 参数的 Bet 在独立的 4,096 道无提示 Chain5 上单次前向答对 3,666 道（89.50%），Chain0–4 各组均超过 99%。本文提供权重、原题、逐题输出和复算入口，并说明怎样从 25 个归档检查点重建 Bet。

**关键词**　递归状态、因果注意力、关系复合、消融、可复现性

## 1 研究问题

给定同一张事实表，查询 $G(x)$ 和 $G(F^4(x))$ 需要选择不同的事实。只凭末词元预测正确答案，与读取整段前缀后组合关系，可以用同题实验分别测量。本文研究两个问题：递归状态怎样进入注意力查询，以及两块都采用该机制时，模型在多步查询中能学到什么。

标准 Transformer 用词元表示的投影构造查询。Stargazer 在读取前插入固定宽度的递归查询状态，采用 GRU 的门控更新。Recurrent Memory Transformer 把记忆词元跨输入段传递。Stargazer 的状态在每个词元更新，并直接用于读取。项目前缀均值版本和保留 Transformer 末端的 Aleph，提供了早期演化记录。[Vaswani 等，2017][attn] · [Cho 等，2014][gru] · [Bulatov 等，2022][rmt] · [早期协议](../RESEARCH.md) · [Aleph 论文](../conversation_memory/training/v0.001Aleph-chain5-wide-mastery90/confirm-796000/REPORT.zh.md)

## 2 Stargazer 的计算

记 $x_t\in\mathbb R^{64}$ 为某一块在位置 $t$ 的输入，$u_t=\operatorname{LN}(x_t)$。

~~~mermaid
flowchart LR
    X["当前词元表示"] --> QV["QV 16 个四维投影"]
    QV --> QK["QK 单查询池化"]
    PREV["上一前缀的 Z"] --> QK
    QK --> GRU["GRU 更新"]
    PREV --> GRU
    GRU --> NOW["当前 32 维 Z"]
    NOW --> READ["4 头因果读取"]
    PFX["当前层可见前缀的键和值"] --> READ
    READ --> OUT["残差与前馈"]
    X --> OUT
~~~

**QV 投影。** 每个词元经过一次 $64\to64$ 仿射投影，输出切成 16 个四维向量 $s_{t,h}$。这些向量来自同一个 $u_t$ 的不同输出坐标。QV 内没有词元间打分或动态路由。Bet 使用当前词元的投影统计，历史由递归状态传递。

**QK 池化及递归更新。** 所有微头共享从四维到 64 维的仿射映射 $K_s,V_s$。一个学习向量 $q_0$ 与上一前缀的状态共同形成池化查询。

$$
\begin{aligned}
q_t &= q_0+W_pZ_{t-1},\\
\alpha_{t,h} &= \operatorname{softmax}_{h}\!\left(q_t^\top K_s(s_{t,h})/\sqrt{64}\right),\\
m_t &= \sum_{h=1}^{16}\alpha_{t,h}V_s(s_{t,h}),\\
Z_t &= \operatorname{GRUCell}(m_t,Z_{t-1}),\qquad Z_0=0.
\end{aligned}
$$

池化的归一化轴是当前词元的 16 个微头。$Z_t\in\mathbb R^{32}$ 对每个前缀生成一次，增量计算时覆盖上一份状态。每层有各自的递归参数和状态。

**KV 读取及残差。** 对每个读取头 $j=1,\dots,4$，以 $Z_t$ 投影出 16 维查询 $Q_j(Z_t)$，以该块的归一化输入 $u_i$ 生成 $K_j(u_i),V_j(u_i)$。对 $i\le t$ 做密集缩放点积注意力。

$$
r_{t,j}=\sum_{i\le t}
\operatorname{softmax}_{i\le t}\!\left(Q_j(Z_t)^\top K_j(u_i)/\sqrt{16}\right)V_j(u_i).
$$

四头拼接并投影得到 $a_t$。残差与前置层归一化的 GELU 前馈更新为 $b_t=x_t+a_t$、$y_t=b_t+\operatorname{FFN}(\operatorname{LN}(b_t))$。前缀键值和块输入残差构成查询状态以外的信息通道。$K_s,V_s$ 和 $K_j,V_j$ 分别服务于微头池化和因果读取，是独立的映射。

**因果性与开销。** 由 $Z_0=0$ 归纳，$Z_t$ 只依赖 $1,\dots,t$，KV 掩码也只允许这些位置。该性质逐层保留。QV 和 KV 的输入投影可以批量执行，GRU 仍有时间依赖。长度为 $T$、隐藏宽度为 $d$ 时，每块的密集读取含 $O(T^2d)$ 计算项。递归块的增量接口只保留一份 $Z$，KV 缓存占用 $O(Td)$。Bet 的验收入口执行完整前缀前向。[核心实现](../recursive_stargazer.py) · [块与缓存实现](../stargazer.py) · [PyTorch SDPA][sdpa]

## 3 Bet 的模型和题面

Bet 是本研究冻结的双递归 Stargazer 检查点。模型词表大小为 32，序列上限和隐藏宽度均为 64。两块的前馈宽度分别是 512 和 145，最后用层归一化和线性词表投影输出分数。完整模型有 175,635 个推理参数。

两块之前还有 Aleph 的情景前置层。它以 32 个二维头读取固定的七条 36 维情景特征。前置层自己的 32 维滚动状态用五维符号码保留当前词元，其余 27 维由 GRU 更新。解码表示加入绝对位置嵌入后进入第一块。这个前置状态与两块内部的 $Z$ 分开。实验中的 $F/G$ 事实随题面输入，情景特征在各题之间固定。本文的消融没有测量情景层的独立贡献。[情景前置实现](../conversation_memory/aleph_episode_front.py) · [Bet 冻结清单](../releases/Bet/manifest.json)

**合成任务。** Chain0 是原始键值查表。对 $k\in\{1,\dots,5\}$，Chain $k$ 询问 $G(F^{k-1}(x))$。$F$ 是八个节点上的排列，$G$ 把节点映射到词表中的 16 种值。题面含随机顺序的 16 条三元组、8 个干扰词元和末尾查询，共 64 个词元。例如，若路径为 $1\to3\to7\to2\to6$，Chain5 求 $G(6)$。

常规题混合随机排列与八节点环，也允许重复的 $G$ 值。严格题要求单个八节点环和互异的 $G$ 值，同图同起点的 Chain1–5 因此有五个不同答案。标准答案由独立解题器重算。验收按完整 $F/G$ 映射排重，忽略三元组顺序、起点和查询深度。正式预测取末位置的词表 argmax，题面不含教师提示和中间标签。[题目生成](../logic_v1/tasks.py) · [参考解题器](../logic_v1/verify.py)

## 4 末端模块的同题消融

以第 796,000 步新版 Aleph 的平均权重为起点。它的末端是因果 Transformer。各替换方案冻结情景前置层、嵌入和第一块递归 Stargazer，使用相同的 2,048 道测试题。拟合读出所用的 65,536 道历史题在常规和严格组之间均分。双 Stargazer 与重训 Transformer 的末端分别有 52,113 和 52,160 个可训练参数。两者都从新初始化的末块和复制来的输出端出发，以 AdamW、批量 512、80 轮训练，由两组平均开发交叉熵选中第 40、12 轮。

门控块先将输入归一化为 $\tilde x$，再计算 $x+W_o[\operatorname{SiLU}(W_g\tilde x+b_g)\odot(W_v\tilde x+b_v)]+b_o$。单层的隐藏宽度为 256，双层每层为 128，末尾均接归一化和线性词表投影。注意力前后的双层读出各自拟合，均有 52,256 个可训练参数。注意力后的读出取原前馈层之前的末词元表示，包含已训练的注意力输出与残差。

| 末端或干预 | 常规正确数 / 1,024 | 严格正确数 / 1,024 | 合计 |
|:--|--:|--:|--:|
| 原 Aleph 末端 Transformer | 906 | 1,001 | 93.12% |
| 末词元线性读出 | 70 | 68 | 6.74% |
| 末词元单层门控 MLP | 74 | 54 | 6.25% |
| 末词元双层门控 MLP | 68 | 56 | 6.05% |
| 保留已训练末端注意力，再接线性层 | 256 | 146 | 19.63% |
| 保留已训练末端注意力，再接双层门控 MLP | 756 | 863 | 79.05% |
| 重新初始化并拟合 Transformer 末端 | 247 | 173 | 20.51% |
| 重新初始化并拟合第二个递归 Stargazer | 432 | 455 | 43.31% |
| 双递归 Stargazer，推理时清零第二层 $Z$ | 138 | 102 | 11.72% |

同样大小的双层门控 MLP，从已训练注意力及其残差之后取表示时，净多答对 1,495 道。线性探针也显示，$F^4(x)$ 在该位置的可读性从 20.80% 提到 61.18%，第一块末词元原本已可线性读出 $F^2(x)$（100%）。该组对比定位了可读出的信息。各节点在网络内如何计算仍有多种可能。

参数接近的两种重训末端相差 467 道，双 Stargazer 高 22.80 个百分点。这个比较保留了以 Transformer 为下游共同训练过的第一块，且只有一条初始化轨迹。清零第二层 $Z$ 的推理干预使得分下降 31.59 个百分点。读取查询的投影偏置、KV 和残差仍在计算。它支持本检查点依赖动态状态，清零也改变了推理时的输入分布。原末端 Transformer 的 93.12% 来自其原有训练经历。[线性消融][linear] · [门控消融和探针][gated] · [匹配末端实验][stack]

## 5 联合训练及 Bet 的独立验收

冻结底座的消融完成后，两块递归 Stargazer 与输出端一起续训。课程逐级加入 Chain0–5，并保留旧题复习。后期用同图较短查询、中间节点、KV 读取位置和乱序配对提供训练信号。情景前置层与词元嵌入保持冻结，后段更新了八个事实后位置的既有位置嵌入。训练用 AdamW，批量 128，每 500 步保存。方法和学习率分阶段变化，续训谱系中也有分支实验。

第一次按每组 256 题的开发集选出截至 295,500 步的平均权重。开发集的常规、严格分数为 91.02%、92.19%，独立题为 86.57%、89.94%。第二次把开发集扩大到各 4,096 道，根据较弱题组锁定模型，再评分一套新题。Bet 由训练编号 388,000 至 400,000 间隔 500 步的 25 份权重平均得到。405,000 步是单独保存的后续训练节点。

| 第二套封存题 | 100,000 步起点 | Bet |
|:--|--:|--:|
| Chain5 常规 | 1,234 / 2,048（60.25%） | 1,825 / 2,048（89.11%） |
| Chain5 严格 | 1,206 / 2,048（58.89%） | 1,841 / 2,048（89.89%） |
| Chain5 合计 | 2,440 / 4,096（59.57%） | 3,666 / 4,096（89.50%） |

Bet 在同题上净增 1,226 道。Chain0–4 各级较弱一组依次为 99.71%、100.00%、99.95%、99.85% 和 99.17%。两组 Chain5 均未超过 90%。本表与第 4 节的题库和训练经历有差别，架构对照应采用第 4 节的同题结果。Bet 属于从已有模型接续的一条谱系。当前实验覆盖八节点、固定词表与 64 词元题面，长上下文效率、异构硬件吞吐和其他规模上的泛化仍待测量。[联合训练起点][joint] · [两次独立验收与分支][continued]

## 6 如何复现

### 6.1 复算冻结 Bet

复现实验使用 Windows、Python 3.15.2、PyTorch 2.15.0+cu132 和 RTX 5090 Laptop GPU。保留项目源码及 releases/Bet 文件夹。在项目根目录打开 PowerShell，先核对当前 Python，再运行冻结复算。

~~~powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m conversation_memory.bet --device cuda
~~~

复算入口校验包内文件的 SHA-256，重新计算 22,528 道题的标签，再逐题比较预测。最后一行应为 **Bet: reference labels and all archived predictions reproduced.** 该入口只依赖项目源码和冻结包。包内情景特征是验收所用的七条快照。权重用 weights_only=True 加载，计算关闭 TF32。在实验机器上，CUDA 和 --device cpu 均复现了全部预测。

### 6.2 从存档重建权重，复算消融

以下命令需要完整本地实验归档。第一条复算 Bet 的 25 检查点均值并逐张量对照冻结权重。后面三条分别复算门控、探针和末端替换的逐题预测。

~~~powershell
$base = 'conversation_memory/training/v0.001Aleph-chain5-wide-mastery90'
$linear = "$base/confirm-transformer-ablation-20261118"
$gated = "$base/confirm-gated-transformer-ablation-20261118"
$stack = "$base/confirm-stargazer-stack-20261118"
$late = "$base/joint-stargazer-chain0-5-20261118/continuation-90/averaged-start-run"
python -m conversation_memory.bet --average-from $late
python -m conversation_memory.aleph_chain5_gated_ablation --prior $linear --output $gated --replay
python -m conversation_memory.aleph_chain5_gated_probes --experiment $gated --replay
python -m conversation_memory.aleph_chain5_stargazer_stack --prior $gated --output $stack --replay
~~~

**重训范围。** 重新拟合两个预算匹配的末端时，去掉最后一条命令的 --replay，把 --output 改为新的目录。四个历史题库、固定抽样种子、原前置层权重、优化器设置和开发集散列均在该实验的 results.json。Bet 的完整训练还依赖早期课程存档。续训入口为 conversation_memory.aleph_stargazer_curriculum，各阶段条件、学习率系数和步数随各分支的 history.json、branch.json 保存。本文复现了保存权重的逐题预测和 Bet 的权重平均，整条训练谱系尚未从随机初始化重新执行。

## 参考文献

1. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). [Attention Is All You Need][attn]. *NeurIPS*, 30.
2. Cho, K., van Merriënboer, B., Gülçehre, Ç., et al. (2014). [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation][gru]. *EMNLP*, 1724–1734.
3. Bulatov, A., Kuratov, Y., & Burtsev, M. S. (2022). [Recurrent Memory Transformer][rmt]. *NeurIPS*, 35.

[attn]: https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
[gru]: https://aclanthology.org/D14-1179/
[rmt]: https://arxiv.org/abs/2207.06881
[sdpa]: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
[linear]: ../conversation_memory/training/v0.001Aleph-chain5-wide-mastery90/confirm-transformer-ablation-20261118/REPORT.zh.md
[gated]: ../conversation_memory/training/v0.001Aleph-chain5-wide-mastery90/confirm-gated-transformer-ablation-20261118/REPORT.zh.md
[stack]: ../conversation_memory/training/v0.001Aleph-chain5-wide-mastery90/confirm-stargazer-stack-20261118/REPORT.zh.md
[joint]: ../conversation_memory/training/v0.001Aleph-chain5-wide-mastery90/joint-stargazer-chain0-5-20261118/REPORT.zh.md
[continued]: ../conversation_memory/training/v0.001Aleph-chain5-wide-mastery90/joint-stargazer-chain0-5-20261118/continuation-90/REPORT.zh.md
