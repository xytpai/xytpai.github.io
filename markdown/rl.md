| [返回主页](index.html) |

---



## 1. Policy Gradient

机器人在环境中的表现为：<br>
采集信息 --> 思考并产生动作 --> 给予反馈 --> 采集信息 --> <br>
**实际上，采集的信息不单单只是环境的信息，还有机器人本身的一些状态；**<br>
**反馈不单单只是环境给予的反馈，还存在机器人自身给自身的反馈。** <br>
先不考虑自身信息及反馈，假设我们的机器人走出这么一条路径：
$$
\tau = 
\{s_1,a_1,s_2,a_2,s_3,a_3...,s_t,a_t\}
$$

其中, s代表机器人观测环境的信息(如图像声音), a代表机器人的动作。我们可以描述机器人走出这条路径的几率：<br>

$$
p_\theta(\tau) = p(s_1)\prod_{t=1}^T
p_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)
$$

其中，p(s) 为环境产生某状态的概率，这个由环境决定。<br>
而 p(s'|s,a) 为环境在上一个状态与机器人的动作后产生的新状态，也由环境决定。<br>
机器人只关心 p(a|s) 也即对其观测信息产生的动作。 <br>
我们认为最终机器人的死亡(终止)取决于环境，是环境杀的机器人，因此最终以一个 p(s'|s,a) 结束。 <br>
除此之外，我们还需要定义反馈 reward(s,a) 它可以是随机性的。<br>
因此我们可以定义一个机器人的总反馈函数： <br>
$$
R_\theta=\sum_\tau reward(\tau)p_\theta(\tau) \\
reward(\tau)=\sum_{i=1}^t reward_\tau(s_i,a_i)
$$

接下来对总反馈函数R求导： <br>
$$
\nabla R_\theta = \sum_\tau reward(\tau)
\nabla p_\theta(\tau) \\
= \sum_\tau reward(\tau)
p_\theta(\tau) \frac{\nabla p_\theta(\tau)}
{p_\theta(\tau)} \\
= \sum_\tau reward(\tau) p_\theta(\tau)
\nabla logp_\theta(\tau) \\
= E_{\tau -p_\theta(\tau)}[reward(\tau)\nabla logp_\theta(\tau)] \\
\approx \frac1N \sum_{n=1}^N
reward(\tau_n)\nabla logp_\theta(\tau_n)\\
=\frac1N \sum_{n=1}^N \sum_{t=1}^{T_n} 
reward(\tau_n) \nabla logp_{\theta}(a_{nt}|s_{nt})
$$

注意这里对**每一条路径的反馈是一个能代表全局评价的反馈**(或每一步反馈的和)。 <br>
可以从公式中直观看出，实际上训练加强了对全局结果影响好的那几个动作。

```python
# 算法执行步骤
1. 尽可能多地采样路径，并计算出每条路径的反馈。
2. 使用这些数据，计算出每个参数的梯度，采用梯度上升优化。
3. 更新参数后，再重复第一步。
```

**几个重要的Tips:**

1. 由于R(tau)的值可能都是正的，可改为 R(tau) - b 偏移最好取E(R(tau))。
2. 将反馈精确到每个动作，某个动作得到的反馈为这个动作之后所有的反馈和：

$$
\nabla R_\theta = \frac1N \sum_{n=1}^N \sum_{t=1}^{T_n}
(\sum_{t'=t}^{T_n}\gamma^{t'-t}
reward_n(t')-b) \nabla logp_\theta(a_{nt}|s_{nt})
, \ \ \ \ \gamma<1
$$

3. 可以用一个优势函数来代替上式中括号内的公式，<br>
   它代表着当前参数下现在在st下执行at相较于其它动作有多好。

$$
A_\theta(s_t,a_t)=\sum_{t'=t}^{T_n}\gamma^{t'-t}
reward_n(t')-b \\
\nabla R_\theta = \frac1N \sum_{n=1}^N \sum_{t=1}^{T_n}
A_\theta(s_t,a_t)
\nabla logp_\theta(a_{nt}|s_{nt})
$$






## 2. Proximal Policy Optimization

看别的模型的结果来更新自己的参数：
$$
\nabla R_\theta = \sum_\tau reward(\tau) p_\theta(\tau)
\nabla logp_\theta(\tau) \\
= \sum_\tau reward(\tau) p_{\theta'}(\tau) \frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)}
\nabla logp_\theta(\tau) \\
= E_{\tau -p_{\theta'}(\tau)}[\frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)}
reward(\tau)\nabla logp_\theta(\tau)] \\
\approx \frac1N \sum_{n=1}^N
\frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)}
reward(\tau_n)\nabla logp_\theta(\tau_n),\ 用另外那个采样\\
= \frac1N \sum_{n=1}^N \sum_{t=1}^{T_n}
\frac{p_{\theta}(a_{nt}|s_{nt})}{p_{\theta'}(a_{nt}|s_{nt})}
\frac{p_{\theta}(s_{nt})}{p_{\theta'}(s_{nt})}
A_{\theta'}(s_t,a_t)
\nabla logp_\theta(a_{nt}|s_{nt}) \\
\approx
\frac1N \sum_{n=1}^N \sum_{t=1}^{T_n}
\frac{p_{\theta}(a_{nt}|s_{nt})}{p_{\theta'}(a_{nt}|s_{nt})}
A_{\theta'}(s_t,a_t)
\nabla logp_\theta(a_{nt}|s_{nt})
$$

我们可以反推出目标函数，同时控制两个模型的差别不要太大：

$$
J^{PPO}_\theta = E_{(s_t,a_t)-\pi_{\theta'}}[
    \frac{p_{\theta}(a_{t}|s_{t})}{p_{\theta'}(a_{t}|s_{t})}
A_{\theta'}(s_t,a_t)]-\beta KL(act(\theta),act(\theta')) \\
J^{PPO2}_\theta = \sum_{(s_t,a_t)}min(
\begin{matrix}
   A_{\theta'}(s_t,a_t)\frac{p_{\theta}(a_{t}|s_{t})}{p_{\theta'}(a_{t}|s_{t})} \\
   A_{\theta'}(s_t,a_t)clip(
\frac{p_{\theta}(a_{t}|s_{t})}{p_{\theta'}(a_{t}|s_{t})},
1-\epsilon,1+\epsilon) \\
\end{matrix}
)
$$

PPO2的clip(a,b,c)的算法为：当a<b时输出b，当a>c时输出c，否则输出a。<br>
其精确意思为：当A>0时，我们希望p_t的值越大越好，但同时与希望与p_t'的比值不超过1+eps;  当A<0时，我们希望p_t的值越小越好，但同时希望与p_t'的比值不小于1-eps。

```python
# 算法执行步骤
1. 初始化一个参数t。
2. 将t模型参数拷贝到t2模型放到环境中采样多次，不断优化原参数t。
3. 重复步骤2。
```





## 3. Q-learning

Q-learning 中我们拟合的是一个评价现在行为好坏的函数。有两种评价方式：

$$
MC: \ \ V_\pi(s_t)=G_t, \ \ 直接估计这个状态最终能得到多少反馈 \\
TD: \ \ V_\pi(s_t)=V_\pi(s_{t+1})+r_t, \ \ 估计两状态间的反馈差
$$

Gt的方差比较大，TD方法需要准确估计V(st+1)。 <br>
Q函数的输入为状态与行为，Q(s,a)。 可以用两种方法建模Q函数，第一种，输入状态与行为输出评价；第二种，输入状态输出每个行为的评价。注意第二种方法行为必须是离散的。

```python
# 基本算法执行步骤（使用TD方法）

1. 初始化估值网络的参数。
2. 将这个估值网络放到环境中做一步互动，得到s1,a1,r1,s2，存入记忆缓存。
   这里，估值网络获得环境输入s1，产生a1动作，得到r1收益，再获得环境输入s2。
   注意，这个估值网络产生动作为：当前最大估计收益对应的动作(大概率)，随机(小概率)。
3. 从记忆缓存中随机采样一批(s1,a1,r1,s2)，令Q_=Q
   更新Q参数使得 Q(s1,a1) ~= r1 + max_a(Q_(s2,a)) 越接近越好。
   后面max_a函数为Q函数输入s2状态时收益最大的那个动作的收益。
   Q_为之前那个估值网络的参数，先固定，训练N步后，将调整后的Q拷贝到Q_，更新。
4. 重复步骤2.
```

![](img\dqntd.jpg)

**几个重要的Tips:**

1. 为解决DQN估计出的收益值比实际偏大的问题（Double DQN）:

$$
Q(s_t,a_t) \to r_t + \max_aQ(s_{t+1},a)，一般操作后边max项会往高估的选 \\
Q(s_t,a_t) \to r_t + Q'(s_{t+1},arg\max_aQ(s_{t+1},a))，改进后选动作的网络与评估分值的网络分离\\
（注意这里argmax不能反向传播，双Q网络更新方法跟DQN类似）
$$

2. 为了更好地训练可以更改DQN的结构（Dueling DQN）： <br>
   下图中，V(s)为标量，后面使用标量与向量逐项相加，其中向量A(s,a)需要经过标准化处理<br>
   处理过程为：将 A(s,a) 的每一项减去平均值使整个向量的和为1。
   

![](img\duelingdqn.jpg)

3. 结合MC与TD方式（Multi-step）：
```python
一次性使用多个数据：(s1,a1,r1, s2,a2,r2, s3,a3,r3, s4)
Q(s1,a1) --> r1+r2+r3 + max_a(Q(s4,a)) 
```

4. 增加参数随机性（Noisy Net）：<br>
   在每一局互动开始时（直到这局结束），将估值网络参数每一个值加上一个高斯随机数。 

5. 估计各个动作Q值的分布（Distributional DQN）：<br>
   每一个动作的输出由很多个bin组成，它们的和为1，得到的估值即为期望。 <br>
   可以通过分析这个动作的分布方差来评估这种决策的稳定性。

![](img\distributionaldqn.jpg)

6. 如果动作是连续的：<br>
   方案一，使用采样离散化；方案二：使用如下网络结构：<br>
   （注意图中的matrix需要一种机制确保它是正定的）

![](img\continuousaction.jpg)





## 4. Actor-Critic

在 Policy Gradient 中我们得出的参数梯度为：

$$
\nabla R_\theta \approx \frac1N \sum_{n=1}^N \sum_{t=1}^{T_n}
(\sum_{t'=t}^{T_n}\gamma^{t'-t}
reward_n(t')-b) \nabla logp_\theta(a_{nt}|s_{nt})
, \ \ \ \ \gamma<1 \\
G_{nt}=\sum_{t'=t}^{T_n}\gamma^{t'-t}
reward_n(t')
$$

可以看到 Gnt 是一个随机量（代表某步之后的有效累积反馈），因为互动是有随机性的。<br>
我们用一个网络来估计这个Gnt的期望，用另一个网络来估计b的期望:
$$
E[G_{nt}]=Q_{\pi}(s_{nt},a_{nt}) \\
E[b]=V_{\pi}(s_{nt}) \\
Q_{\pi}(s_{nt},a_{nt})=reward_n(t)+V_\pi(s_{n[t+1]})
$$

上面公式中我们希望V是Q的期望值，用如下式子简写，只用一个网络：

$$
\sum_{t'=t}^{T_n}\gamma^{t'-t}
reward_n(t')-b \\= Q_\pi(s_{nt},a_{nt})-V_\pi(s_{nt}) \\
=reward_n(t)+V_\pi(s_{n[t+1]})-V_\pi(s_{nt})
$$

**最终得到 Advantage Actor-Critic：**

$$
\nabla R_\theta \approx \frac1N \sum_{n=1}^N \sum_{t=1}^{T_n}
(reward_n(t)+V_\pi(s_{n[t+1]})-V_\pi(s_{nt}))
\nabla logp_\theta(a_{nt}|s_{nt}) \\
Advantage Function\to reward_n(t)+V_\pi(s_{n[t+1]})-V_\pi(s_{nt})
$$

```python
# 算法执行步骤

1. 用策略网络跟环境做互动，得到s,a,r序列。
2. 使用得到的序列来估计优势函数网络 (advantage function)
3. 套用上述公式去更新策略网络（固定优势函数网络）。
4. 重复步骤1.
```
**Advantage Actor-Critic 的几个Tips：**

1. 引入一些机制使动作输出分布的熵大一点。

2. 引入一些共享结构：

![](img\actor-critic-shared.jpg)

**Asynchronous Advantage Actor-Critic（A3C）：**

```python
# 算法执行步骤

0. 主设备存储全局参数，n台从设备同时工作。
1. 主设备将全局参数拷贝到从设备中。
2. 从设备并行与环境做互动并计算各自的梯度。
3. 从设备将参数的更新增量发给主设备（异步）。
4. 主设备把这些更新增量加起来更新全局参数。
5. 重复步骤1.
```

**Pathwise Derivative Policy Gradient：**

![](img\Pathwise Derivative Policy Gradient.jpg)

```python
# 基本算法执行步骤（使用TD方法）

1. 初始化Actor,Q的参数。
2. 根据Actor的行为在环境中做互动得到s1,a1,r1,s2，存入记忆缓存。
   这里，估值网络获得环境输入s1，产生a1动作，得到r1收益，再获得环境输入s2。
   注意，这个估值网络产生动作为：当前最大估计收益对应的动作(大概率)，随机(小概率)。
3. 从记忆缓存中随机采样一批(s1,a1,r1,s2)，令Q_=Q
   更新Q参数使得 Q(s1,a1) ~= r1 + Q_(s2,Actor(s2)) 越接近越好。
   这里Actor参数先固定住，Q_为之前那个估值网络的参数，也先固定住。
   训练N步后，将调整后的Q拷贝到Q_，将上式的Q固定住调整Q_与Actor的参数。
4. 重复步骤2.
```





## 5. Sparse Reward

**如何设计收益**，下面有一个FPS游戏的反馈参考：

```python
活着        -0.008
掉血        -0.05
子弹消耗     -0.04
捡到医疗包   +0.04
捡到子弹     +0.15
呆在原地     -0.03
运动        +9e-5
```

**好奇心机制：**

![](img\curiosity.jpg)

上图中的网络输出的是好奇心机制的反馈， <br>
输入当前动作 at 以及当前状态 st ，预测下一个状态，这个预测与s[t+1]差别越大收益越大。<br>
这里的 FeatureExt 用来过滤状态的无用信息（如FPS游戏里的风吹草动）。<br>
为了训练 FeatureExt 引入 Network2 输出为进行的动作预测，与at越像越好。

**进阶学习机制：**

如让机器臂把板子放到柱子上，可以由近及远，让难度不断上升。
```python
1. 确定一个目标状态（比如机器手臂抓到东西）。
2. 在这个目标状态周围找其他的状态（还没抓到东西快抓到了）。
3. 用这些周围状态做互动看能不能达到目标状态，又采样到其他状态。
4. 将其他状态中反馈极端（太简单或太难）的状态去掉。
5. 再根据这些其他状态采样到更多的状态。
6. 依照层次顺序制定进阶学习策略。
```

**层级式学习机制：**

将终极目标拆分成其他小目标，再拆分成更小的目标，形成一个愿景树。





## 6. Imitation Learning

如果没有办法从环境得到明确的反馈，可以让机器人模仿专家的行为。

**直接模仿专家行为：**

完全模仿专家行为，当作有监督学习。<br>
但是，专家的行为不太会出错，因此机器人无法学到如何处理特殊情形。

**专家指导行为：**

在一轮互动中机器人完全按照自己的决策做动作。<br>
同时，由专家给出某个状态应该产生的动作，但这个不影响机器人行为。<br>
等到数据采样完毕后机器人再调整参数来学习到专家的策略。

**反向强化学习（IRL）：**

![](img\IRL.jpg)

```python
# 基本算法执行步骤

1. 专家与机器人都与环境进行一轮互动，进行采样。
2. 训练收益网络，输入为专家与机器人的数据，收益网络的输出为一个实数，
   使专家的收益尽可能比机器人的高。
3. 把这个收益网络当作Reward用一般的强化学习方法训练机器人。
4. 训练后的机器人与环境进行一次互动，进行采样。
5. 重复步骤2.
```

**第三人称模仿学习：**

机器人如果看到别人的动作来联想到自己的动作。 <br>
一般的模仿学习是手把手教，这个是通过用眼睛看别人的动作学。 <br>
可以使用生成对抗网络（GAN），采集两种图像，一个是第一人称视角，一个是第三人称视角。<br>
使用第三人称视角图像经过生成网络生成第一人称视角图像。

**反向强化学习与 GAN 的联系：**

经典 GAN 中，我们期望生成一些逼真的图像，实际上也是对真实图像的模仿学习。<br>
机器人与专家的互动数据，训练一个 *判别器* 来输出他们的收益之差，期望专家收益越大越好。<br>
然后用收益网络训练机器人，机器人再与环境互动产生新数据，这个相当于 *生成器* 。<br>
其实质是期望让机器人生成一些逼真的行为。





---

内容出处：https://www.bilibili.com/video/av24724071