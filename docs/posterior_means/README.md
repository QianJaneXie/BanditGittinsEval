# 两种 posterior mean：latent arm mean 与完整测试集均值

本项目中需要区分两个估计对象：潜在的总体均值参数，以及当前这份固定测试集的完整均值。它们都可以取 posterior mean，但得到的数值和不确定性不同。

**Posterior mean 和 posterior expected mean 本身是同一个概念，都是后验期望。这里的区别在于对哪个随机变量求期望：`E[theta | D]` 还是 `E[F | D]`。**

| 名称 | 估计对象 | 记号 | 回答的问题 |
|---|---|---|---|
| Latent posterior mean | 生成每题得分的潜在均值参数 | $\mu_n=\mathbb{E}[\theta\mid D_n]$ | 在模型假设下，这个 arm 的总体平均表现是多少？ |
| Full-test-set posterior mean | 当前固定测试集全部题目的实际平均得分 | $M_n=\mathbb{E}[F\mid D_n]$ | 已经看到部分结果后，这份完整测试集最终的均分预计是多少？ |

本项目的 simple regret 用完整矩阵的实际行均值计算，因此当前 Gittins 的推荐、acquisition 和停止判断统一使用第二个目标。Latent posterior 仍作为预测未评估题目得分的模型。

## 1. 模型与记号

先只考虑一个 arm，省略 arm 下标。设固定测试集有 $N$ 道题，每题的得分为 $Y_i$。当前已经评估 $n$ 道题，其观测集合为 $D_n$，得分总和为 $S_n$，剩余题数为 $m=N-n$。

当前实现采用 normal-normal 工作模型：

$$
\theta\sim\mathcal{N}(\mu_0,v_0),\qquad
Y_i\mid\theta\sim\mathcal{N}(\theta,\tau_{\mathrm{cell}}^2).
$$

给定 $\theta$ 后，各题得分在这个模型下条件独立。这里 $v_0$ 和 $\tau_{\mathrm{cell}}^2$ 都是 **variance**；对于 correctness 等二元得分，Gaussian 是当前实现使用的近似模型。

如果一次评估 $B$ 道题，batch mean 的 observation variance 是 $\tau_{\mathrm{cell}}^2/B$。代码在 `batch_observation_model=True` 时，会把传入的 batch-mean variance 乘以 $B$，再与逐题的观测计数配合使用。默认的 $1/(4B)$ 对应 $\tau_{\mathrm{cell}}^2=1/4$。

## 2. Latent posterior mean：估计 theta

根据已观测数据，潜在参数的后验为

$$
\theta\mid D_n\sim\mathcal{N}(\mu_n,v_n),
$$

$$
v_n=\left(\frac{1}{v_0}+\frac{n}{\tau_{\mathrm{cell}}^2}\right)^{-1},\qquad
\mu_n=v_n\left(\frac{\mu_0}{v_0}+\frac{S_n}{\tau_{\mathrm{cell}}^2}\right).
$$

令 $\kappa=\tau_{\mathrm{cell}}^2/v_0$，也可以写成

$$
\mu_n=\frac{S_n+\kappa\mu_0}{n+\kappa}.
$$

它综合了观测数据和 prior。即使已经评估完当前测试集，有限的 $N$ 次观测通常也不能让潜在参数 $\theta$ 完全确定，因此 $v_N>0$，而 $\mu_N$ 通常仍不等于完整测试集的 empirical mean。

这对估计 $\theta$ 是正常的后验收缩；问题在于不能把它直接当作已完全揭晓的测试集均值。

## 3. Full-test-set posterior mean：估计 F

我们实际评估的目标是这份固定测试集的完整均值：

$$
F=\frac{1}{N}\sum_{i=1}^{N}Y_i.
$$

在离线实验中，完整矩阵的这些得分已经固定；算法只能看到被揭示的 cells。这里的 posterior 描述算法对尚未揭示结果的不确定性，每次评估都是揭示已有结果。

已观测题目的得分直接使用已知数值，未观测题目的条件期望由 $\mu_n$ 给出，因此

$$
M_n=\mathbb{E}[F\mid D_n]
=\frac{S_n+(N-n)\mu_n}{N}.
$$

当 $n>0$ 时，令 $\bar Y_n=S_n/n$，就得到讨论中的加权形式：

$$
M_n=\frac{n}{N}\bar Y_n+\frac{N-n}{N}\mu_n.
$$

所以这里的 empirical mean 是**已评估部分**的均值，latent posterior mean 只用于预测剩余部分。代码使用求和形式，避免 $n=0$ 时出现未定义的 empirical mean。

对应的 posterior variance 为

$$
V_n=\operatorname{Var}(F\mid D_n)
=\frac{(N-n)^2v_n+(N-n)\tau_{\mathrm{cell}}^2}{N^2}.
$$

第一项来自潜在均值的不确定性，第二项来自尚未揭示的题目得分在给定 $\theta$ 后的变动。**不能只把 latent std 乘以剩余题目比例**，否则会漏掉第二项。

| 评估状态 | Latent mean 与 variance | Full-test-set mean 与 variance |
|---|---|---|
| 未评估，$n=0$ | $\mu_0,\ v_0$ | $\mu_0,\ v_0+\tau_{\mathrm{cell}}^2/N$ |
| 部分评估，$0<n<N$ | $\mu_n,\ v_n$ | $(S_n+(N-n)\mu_n)/N,\ V_n$ |
| 全部评估，$n=N$ | $\mu_N,\ v_N>0$ | $S_N/N,\ 0$ |

Completed arm 的 $F$ 已经完全知道，因此其 finite posterior mean 精确等于完整 empirical mean，finite posterior std 为零。这不表示其 latent 参数也已经完全知道。

Finite variance 也并非在所有状态下都小于 latent variance：未评估时它多出 $\tau_{\mathrm{cell}}^2/N$。

## 4. 一个可以手算的例子

为便于手算，取 $N=10$、$\mu_0=0.5$、$v_0=0.25$、$\tau_{\mathrm{cell}}^2=0.25$。这些是示例参数，不是 GSM8K 实验的 prior。

假设同一个 arm 的前 4 道题答对 3 道，全部 10 道题最终答对 7 道：

| 状态 | 已观测 empirical mean | Latent mean $\mu_n$ | Full-test-set mean $M_n$ | Latent variance $v_n$ | Finite variance $V_n$ |
|---|---:|---:|---:|---:|---:|
| $n=0$ | 未定义 | 0.5 | 0.5 | 0.25 | 0.275 |
| $n=4,\ S_n=3$ | 0.75 | 0.70 | 0.72 | 0.05 | 0.033 |
| $n=10,\ S_n=7$ | 0.70 | 0.681818 | 0.70 | 0.022727 | 0 |

部分评估时，完整均值的估计为 $(3+6\times0.70)/10=0.72$。完成后，完整测试集得分已经是 0.70；latent estimate 仍受到 prior 0.5 的影响，为 $7.5/11\approx0.681818$。

## 5. 为什么更改均值后，mean-only 推荐可能完全不变？

把 $S_n=(n+\kappa)\mu_n-\kappa\mu_0$ 代入 $M_n$，得到

$$
M_n=a\mu_n+(1-a)\mu_0,\qquad
a=1+\frac{\tau_{\mathrm{cell}}^2}{Nv_0}>0.
$$

如果所有 arms 共享 $N$、$\mu_0$、$v_0$ 和 $\tau_{\mathrm{cell}}^2$，它们就共享同一个严格递增的仿射变换。因此，**在相同观测数据上，mean-only 推荐的排序在精确算术下完全相同**，即使各 arm 已评估的题数不同。

这个结论仅比较同一状态下的推荐排序：

- 它不表示两种 mean 数值相等，也不表示两种 variance 相等。
- 它不保证使用不同 arm-specific priors、噪声或测试集大小时仍保持排序。
- 它不保证更改 acquisition 后还能得到相同数据和推荐轨迹；浮点计算在接近并列时也可能影响数值排序。

未评估 arm 的两种 mean 都仍是 prior mean。因此，改成 finite mean 不会自动解决 general prior 过于乐观时推荐未评估 arm 的问题。

## 6. LCB 应该配合哪一种 std？

均值和 std 应描述同一个目标：

$$
\mathrm{LCB}^{\mathrm{latent}}_k=\mu_k-\lambda\sqrt{v_k},\qquad
\mathrm{LCB}^{F}_k=M_k-\lambda\sqrt{V_k}.
$$

本项目关心完整测试集表现，因此可选 LCB 使用 $M_k-\lambda\sqrt{V_k}$。Completed arm 的 score 自动回到完整 empirical mean；用仍然为正的 latent std 继续惩罚它，与这个 finite target 不匹配。

取 $\lambda=1$，上面的例子在 $n=4$ 时，latent LCB 为 $0.70-\sqrt{0.05}\approx0.476393$，finite LCB 为 $0.72-\sqrt{0.033}\approx0.538341$。完成时 finite LCB 就等于 0.70。

LCB 排序没有 mean-only 的不变性。令 $r_k=(N-n_k)/N$，在共享参数的模型下有 $V_k=a r_k v_k$，因此 finite LCB 的排序等价于比较

$$
\mu_k-\lambda\sqrt{\frac{r_k}{a}}\sqrt{v_k}.
$$

相对于 latent mean 的尺度，有效 std penalty 会随 arm 接近完成而减小；所有 arms 都使用固定 latent std penalty 的旧规则则没有这个性质。

**目标更一致不等于必须使用 LCB。** 如果相信当前 posterior，并以 posterior expected simple regret 为终止推荐目标，那么

$$
\mathbb{E}[\max_j F_j-F_k\mid D]
=\mathbb{E}[\max_j F_j\mid D]-M_k.
$$

第一项与所推荐的 $k$ 无关，所以最优终止推荐是最大化 $M_k$。LCB 是额外加入的保守推荐选择，效果需由实验判断。这个结论讨论的是给定数据后的最终推荐，不是全局采样或停止策略的最优性。

当前默认 `--recommendation-std-penalty 0` 使用 finite mean；设为 `1` 使用 finite mean minus one finite posterior std。记录的 `recommended_mean` 仍是未减去 penalty 的均值。

## 7. 更改 acquisition 后，是否需要重跑实验？

只替换推荐均值，且满足前面的共享参数条件时，同一轨迹上的 mean-only 推荐排序保持不变。统一修改 Gittins acquisition 则还需要更改 DP 的状态转移。

因为 $M_n=a\mu_n+(1-a)\mu_0$，finite mean 的每次更新增量标准差是 latent mean 更新增量标准差的 $a$ 倍。它与后验剩余总不确定性 $\sqrt{V_n}$ 是不同的量。

在相同有限 horizon、观测安排和精确 DP 下，若评估成本为 $c$，有

$$
\Gamma^F(c)=a\,\Gamma^{\mathrm{latent}}(c/a)+(1-a)\mu_0.
$$

$c/a$ 表示整个后续成本序列都除以 $a$。当前实现保持配置中的 cost 数值不变，使用 finite mean 和相应的 transition std 重新计算 roots；completed index 精确等于 empirical mean。因此，**即使不开 LCB，采样和 natural stopping 也可能改变**，旧实验不能直接当作新 acquisition 的结果。

在已有 GSM8K seed 0、两种 prior 的对照中，两种 prior 的采样轨迹确实都发生变化。这里引用已经完成的实验，详细数学与结果见 [统一 finite-target Gittins 说明](../finite_population_gittins.md)。

## 8. 当前代码如何对应这两个定义？

| 接口或记录 | 当前含义 |
|---|---|
| `posterior_moments` / `posterior_means` | 保留 latent posterior 的 $\mu_n,v_n$ / $\mu_n$ |
| `finite_population_posterior_moments` | 完整测试集的 $M_n,V_n$ |
| `posterior_incumbent` | 按 finite mean 推荐，支持可选 finite std penalty |
| `gittins_index_exploration` / `gittins_post_pull_update` | 使用 finite-target index；返回的 posterior means 也是 $M_n$ |
| `compute_finite_population_roots_lookup_table` | 为 finite mean 的转移过程构建 DP roots |
| `recommended_mean` / `posterior_mean_pulled` | 当前 Gittins runner 记录未惩罚的 finite mean |

对应源码：[posterior 与推荐](../../src/simple_regret_recommend.py)、[Gittins policy](../../src/gittins_policy.py)、[transition 与 roots](../../src/gittins_shrinking_posterior.py)。保留 latent helper 的名字是为了明确两个估计对象；不能仅根据函数名里有 `posterior` 就判断它是哪一种 mean。

读取历史实验时还需要检查版本与元数据：

| 版本 | 推荐目标 | Acquisition / natural stopping 目标 |
|---|---|---|
| `da2b7e6` | Latent mean，可选 latent LCB | Latent mean |
| `f4833e4` | Finite mean，可选 finite LCB | 仍是 latent mean |
| `b667d40` 起 | Finite mean，可选 finite LCB | 统一为 finite mean |

新 Gittins runs 记录 `gittins_index_target=finite_population_mean`，recommendation rule 为 `finite_population_posterior_mean` 或 `finite_population_posterior_mean_minus_std`。在本仓库历史结果的读取约定中，缺少 index-target 字段表示旧的 latent acquisition；旧 `posterior_mean_pulled` 也应按当时版本解释。

已有的 [推荐规则及历史实验](../finite_population_recommendation.md) 和 [统一 acquisition 的实验](../finite_population_gittins.md) 分别保留两次修改的结果，避免把“只改推荐”与“同时改采样”混为一谈。
