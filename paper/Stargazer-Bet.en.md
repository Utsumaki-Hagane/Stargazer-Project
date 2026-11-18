# Stargazer: Recurrent State for Causal Retrieval

**The two-block Bet prototype, Chain5 ablations, and a reproduction protocol**  
Research draft · 18 November 2026 · Maki (penname) & GPT-6.1 Sol

## Abstract

Stargazer is a causal sequence block that projects each token into parallel micro-heads, updates a recurrent 32-dimensional state $Z_t$, and uses that state to query the keys and values of the visible prefix. We call the fixed-width query interface the global conclusion bottleneck. Prefix keys and values and the residual stream remain available to the block. With an Aleph stem frozen, linear and gated MLP readouts from the first block's final-token representation achieved about 6% on Chain5. Feeding the same two-block gated architecture from the output of the trained terminal attention raised accuracy to 79.05%. With matched training budgets and similar parameter counts, a newly fitted recursive Stargazer tail achieved 43.31% and a newly fitted Transformer tail achieved 20.51%. After joint training of the two recursive blocks, the 175,635-parameter Bet model answered 3,666 of 4,096 independent Chain5 questions correctly (89.50%) in one unhinted forward pass. All Chain0–4 groups exceeded 99%. We provide weights, evaluation questions, per-question predictions, and commands to reproduce inference and reconstruct Bet from 25 archived checkpoints.

**Keywords** · recurrent state, causal attention, relational composition, ablation, reproducibility

## 1. Research question

Given the same fact table, $G(x)$ and $G(F^4(x))$ require different lookups. Evaluating different readouts on the same questions lets us compare the information available in the final-token representation with what a learned reader can extract from the full prefix. We ask how a recurrent state can drive an attention query and what a two-block stack using this mechanism can learn on a multi-step relational task.

A standard Transformer projects token representations into queries. Stargazer inserts a fixed-width recurrent query state and updates it with a GRU. Recurrent Memory Transformer carries memory tokens across input segments. Stargazer updates its state at each token and uses it directly for retrieval. The project's earlier prefix-mean prototype and the Aleph model with a terminal Transformer describe the preceding designs. [Vaswani et al., 2017][attn] · [Cho et al., 2014][gru] · [Bulatov et al., 2022][rmt] · [Early protocol](../RESEARCH.md) · [Aleph paper](../conversation_memory/training/v0.001Aleph-chain5-wide-mastery90/confirm-796000/REPORT.zh.md)

## 2. Stargazer computation

Let $x_t\in\mathbb R^{64}$ be a block's input at token position $t$, and let $u_t=\operatorname{LN}(x_t)$.

~~~mermaid
flowchart LR
    X["Current token representation"] --> QV["QV: 16 four-dimensional projections"]
    QV --> QK["QK: single-query pooling"]
    PREV["Previous prefix state Z"] --> QK
    QK --> GRU["GRU update"]
    PREV --> GRU
    GRU --> NOW["Current 32-dimensional Z"]
    NOW --> READ["Four-head causal retrieval"]
    PFX["Visible prefix keys and values"] --> READ
    READ --> OUT["Residual and feed-forward updates"]
    X --> OUT
~~~

**QV projection.** A $64\to64$ affine projection produces 16 four-dimensional vectors $s_{t,h}$. Each vector takes a different group of output coordinates from the projection of $u_t$. QV performs no scoring across token positions or dynamic routing. Bet uses the current token's projections and carries history in the recurrent state.

**QK pooling and recurrence.** All micro-heads share affine maps $K_s,V_s$ from four to 64 dimensions. A learned vector $q_0$ and the previous prefix state form the pooling query.

$$
\begin{aligned}
q_t &= q_0+W_pZ_{t-1},\\
\alpha_{t,h} &= \operatorname{softmax}_{h}\!\left(q_t^\top K_s(s_{t,h})/\sqrt{64}\right),\\
m_t &= \sum_{h=1}^{16}\alpha_{t,h}V_s(s_{t,h}),\\
Z_t &= \operatorname{GRUCell}(m_t,Z_{t-1}),\qquad Z_0=0.
\end{aligned}
$$

The softmax runs over the current token's 16 micro-heads. Each prefix produces one $Z_t\in\mathbb R^{32}$. Incremental execution replaces the previous state with the current one. The two blocks have separate recurrent parameters and states.

**KV retrieval and residuals.** Each read head $j=1,\dots,4$ projects $Z_t$ to the 16-dimensional query $Q_j(Z_t)$. Its keys and values $K_j(u_i),V_j(u_i)$ come from the block's normalized input at each visible position. It applies dense scaled dot-product attention over $i\le t$.

$$
r_{t,j}=\sum_{i\le t}
\operatorname{softmax}_{i\le t}\!\left(Q_j(Z_t)^\top K_j(u_i)/\sqrt{16}\right)V_j(u_i).
$$

Concatenating and projecting the four heads gives $a_t$. The residual and pre-normalized GELU feed-forward updates are $b_t=x_t+a_t$ and $y_t=b_t+\operatorname{FFN}(\operatorname{LN}(b_t))$. The prefix keys and values and the input residual carry information outside the query state. The pooling maps $K_s,V_s$ and the causal read maps $K_j,V_j$ have separate parameters.

**Causality and cost.** Starting from $Z_0=0$, induction gives that $Z_t$ depends only on positions $1,\dots,t$. The KV mask allows the same positions, so causality is preserved through the stack. QV and KV input projections can be batched, while the GRU update has a temporal dependency. For sequence length $T$ and hidden width $d$, dense retrieval contributes an $O(T^2d)$ term per block. The incremental recursive block retains one copy of $Z$ and an $O(Td)$ KV cache. Bet's evaluation uses a full-prefix forward pass. [Recursive block](../recursive_stargazer.py) · [Block and incremental cache](../stargazer.py) · [PyTorch SDPA][sdpa]

## 3. Bet and the evaluation inputs

Bet is the frozen checkpoint with two recursive Stargazer blocks. It has a vocabulary of 32 symbols, a context of 64 tokens, and a hidden width of 64. The two feed-forward widths are 512 and 145. Layer normalization and a linear vocabulary projection produce the final scores. The complete model has 175,635 inference parameters.

Both blocks follow Aleph's episodic front end. It uses 32 two-dimensional query heads to read seven fixed 36-dimensional episodic features. Its own 32-dimensional rolling state allocates five coordinates to the current token's symbol code. A GRU updates the remaining 27 coordinates. The decoded representation receives an absolute position embedding before entering the first block. This front-end state is separate from the two Stargazer states. Each evaluation question supplies its own $F/G$ facts, and the episodic snapshot is shared across questions. The ablations in this paper do not isolate the front end's contribution. [Episodic front end](../conversation_memory/aleph_episode_front.py) · [Bet manifest](../releases/Bet/manifest.json)

**Synthetic tasks.** Chain0 is the original key–value lookup task. For $k\in\{1,\dots,5\}$, Chain $k$ asks for $G(F^{k-1}(x))$. $F$ permutes eight nodes, and $G$ maps each node to one of 16 value symbols. The 64-token input contains 16 shuffled triples, eight distractor tokens, and the query. For example, if the path is $1\to3\to7\to2\to6$, Chain5 asks for $G(6)$.

Regular questions mix arbitrary permutations and single eight-node cycles and permit repeated $G$ values. Strict questions use an eight-node cycle and distinct $G$ values. As a result, Chain1–5 have different answers on the same strict graph and start node. An independent solver recomputes labels. Evaluation excludes duplicate full $F/G$ mappings regardless of triple order, start node, or query depth. The prediction is the vocabulary argmax at the last position of the original prompt. The evaluation prompt contains neither teacher hints nor intermediate labels. [Task generator](../logic_v1/tasks.py) · [Reference solver](../logic_v1/verify.py)

## 4. Paired terminal-block ablations

The ablations start from the newer Aleph model averaged at step 796,000, whose terminal block is a causal Transformer. Replacements freeze the episodic front end, embeddings, and first recursive Stargazer block. All variants use the same 2,048 test questions. The 65,536 archived training questions are evenly divided between regular and strict cases. The Stargazer and Transformer replacement tails have 52,113 and 52,160 trainable parameters, respectively. Both initialize a new terminal block, copy the original output layers, and train with AdamW for 80 epochs at batch size 512. Mean cross-entropy across the two development groups selected epochs 40 and 12, respectively.

For the gated readouts, a block normalizes its input to $\tilde x$ and computes $x+W_o[\operatorname{SiLU}(W_g\tilde x+b_g)\odot(W_v\tilde x+b_v)]+b_o$. The one-block variant has hidden width 256, and the two-block variant has width 128 in each block. Both end in normalization and a linear vocabulary projection. The two-block readers on either side of the attention stage were fitted separately and have 52,256 trainable parameters each. The post-attention reader receives the final-token representation just before the original feed-forward sublayer, including trained attention and its residual.

| Terminal stage or intervention | Regular correct / 1,024 | Strict correct / 1,024 | Overall |
|:--|--:|--:|--:|
| Original Aleph Transformer tail | 906 | 1,001 | 93.12% |
| Final-token linear readout | 70 | 68 | 6.74% |
| Final-token gated MLP, one block | 74 | 54 | 6.25% |
| Final-token gated MLP, two blocks | 68 | 56 | 6.05% |
| Trained terminal attention followed by a linear readout | 256 | 146 | 19.63% |
| Trained terminal attention followed by a two-block gated MLP | 756 | 863 | 79.05% |
| Reinitialized and fitted Transformer tail | 247 | 173 | 20.51% |
| Reinitialized and fitted recursive Stargazer tail | 432 | 455 | 43.31% |
| Two recursive blocks, second-block $Z$ zeroed at inference | 138 | 102 | 11.72% |

With the matched two-block gated reader, taking the representation after trained attention and its residual increased the net count by 1,495 correct answers. A linear probe for $F^4(x)$ improved from 20.80% to 61.18% at the latter position. The first block's final-token representation already supported a linear $F^2(x)$ readout at 100%. The probes measure access to labeled information through a particular readout. Several internal computations can produce the same probe results.

The newly fitted Stargazer tail answered a net 467 more questions than the similarly sized Transformer tail, a 22.80-point difference. Both tails used a frozen stem originally trained with the Transformer downstream, and the comparison uses one initialization trajectory. During inference, zeroing the second block's $Z$ reduced accuracy by 31.59 points. The query projection bias, KV projections, and residuals remained active. The intervention measures dependence on the dynamic state in this checkpoint and shifts the inference distribution. The original Transformer tail's 93.12% reflects its earlier training history. [Linear ablation][linear] · [Gated ablations and probes][gated] · [Matched replacement tails][stack]

## 5. Joint training and independent confirmation

After the frozen-stem ablations, both recursive blocks and the output layers were trained jointly. The curriculum introduced Chain0–5 in order and retained practice on earlier tasks. Later stages used shorter equivalent queries, intermediate-node labels, KV location supervision, and pairs with shuffled facts. The episodic front end and token embeddings remained frozen. Eight existing position embeddings after the facts were updated in later stages. Training used AdamW, a batch size of 128, and a save interval of 500 updates. The schedule included multiple training interventions and branches.

The first confirmation selected an average ending at step 295,500 using 256 development questions per group. Its development scores were 91.02% regular and 92.19% strict. On independent questions, the scores were 86.57% and 89.94%. We then expanded development to 4,096 questions per group, selected by the lower of the two group scores, and locked the candidate before scoring a new test set. Bet averages 25 checkpoints spaced 500 steps apart between training steps 388,000 and 400,000. Step 405,000 is the separately saved restart checkpoint.

| Second held-out suite | Step 100,000 | Bet |
|:--|--:|--:|
| Chain5 regular | 1,234 / 2,048 (60.25%) | 1,825 / 2,048 (89.11%) |
| Chain5 strict | 1,206 / 2,048 (58.89%) | 1,841 / 2,048 (89.89%) |
| Chain5 overall | 2,440 / 4,096 (59.57%) | 3,666 / 4,096 (89.50%) |

Bet gained a net 1,226 correct answers on these same questions. The lower group scores for Chain0–4 were 99.71%, 100.00%, 99.95%, 99.85%, and 99.17%. Both Chain5 groups fell below the target of more than 90%. Section 4 provides the paired architecture comparison. Bet extends one existing training lineage, and this evaluation covers eight-node graphs, a fixed vocabulary, and 64-token inputs. Long-context efficiency, heterogeneous hardware throughput, and generalization to other graph sizes remain open. [Joint-training starting point][joint] · [Confirmation and branch history][continued]

## 6. Reproduction

### 6.1. Replay frozen Bet

The recorded environment was Windows, Python 3.15.2, PyTorch 2.15.0+cu132, and an RTX 5090 Laptop GPU. Keep the project source and the releases/Bet directory. From the project root in PowerShell, check the environment and run the replay.

~~~powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m conversation_memory.bet --device cuda
~~~

The verifier checks SHA-256 digests, recomputes the labels for 22,528 questions, and compares predictions question by question. It prints **Bet: reference labels and all archived predictions reproduced.** on success. This path uses the project source and the frozen bundle, including the seven episodic vectors used at evaluation. We load weights with weights_only=True and disable TF32. Both CUDA and --device cpu matched all archived predictions on the experiment machine.

### 6.2. Reconstruct weights and replay ablations

The following commands use the local experiment archive. The first reconstructs Bet from the 25 saved checkpoints and checks tensor equality. The remaining commands replay the gated readouts, intermediate-node probes, and terminal-block replacements.

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

**Retraining scope.** With the full training archive, rerun the matched-tail fit by dropping --replay from the final command and using a new --output directory. The corresponding results.json records the four training banks, sampling seed, episodic source weights, optimizer settings, and development-set digests. The complete Bet trajectory also depends on earlier curriculum artifacts. The continuation entry point is conversation_memory.aleph_stargazer_curriculum, and each branch's history.json and branch.json record its conditions, learning-rate multipliers, and steps. This paper verifies archived predictions and Bet's checkpoint average. A fresh run of the complete training lineage from random initialization remains untested.

## References

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
