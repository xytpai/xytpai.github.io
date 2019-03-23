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
\frac{p_{\theta}(a_{t}|s_{t})}{p_{\theta'}(a_{t}|s_{t})}
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




