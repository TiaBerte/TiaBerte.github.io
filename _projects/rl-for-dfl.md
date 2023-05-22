---
layout: page
title: RL for Decision Focused Learning
description: Analysis of Soft Actor-Critic algorithm in the context of Decision Making
img: assets/img/sac_comparison.png
importance: 1
category: work
---

# Project description
 This project contains an analysis of the soft actor-critic algorithm applied in the field of decision-focused learning for solving the set multicover problem. We experimented a variation using prioritized experience replay for improving convergence. The algorithm and the environments were mainly derived from the garage library and can be found on [GitHub](https://github.com/TiaBerte/rl-for-dfl).


# Table of Contents
1. [Decision Focused Learning](#section1)
2. [Set Multicover Problem](#section2)
3. [Soft Actor Critic](#section3)
4. [Prioritized Experience Replay](#section4)
5. [Experiments](#section5)
6. [Conclusions](#section6)


# Decision Focused Learning  <a name="section1"></a>
Decision-making algorithms usually require predictive models and a combinatorial optimization phase, however, these two tasks are often tackled independently following the so-called "Predict and Optimize" paradigm which consists in training a machine learning model to maximize the prediction accuracy, and then using the prediction as input for the optimization phase. However, the loss function used to train the model may easily be misaligned with the end goal, which is to make the best decisions possible.    
[Decision-focused learning frameworks](https://arxiv.org/pdf/1809.05504.pdf) integrates prediction and optimization into a single end-to-end system whose predictive model is trained in such a way its predictions are optimal for the decision algorithm.  
The general optimization problem can be expressed as:
<p align="center">
$$
   \underset{\omega}{\text{argmin}} \left\{ \sum_{i=1}^m c(z^*(y_i), \hat{y}_i) \mid y = f(\hat{x}, \omega) \right\}
$$
</p>

where:  
* $$\hat{x}$$ and $$\hat{y}$$ are input-output pairs drawn from some unknown distribution;
* $$y$$ the predictions of the model;  
* $$\omega$$ the parameters of a machine learning model;  
* $$z$$ the actions selected by the decision algorithm;  
* $$c$$ the cost function.  
  
Even if not explicitly presented, $$z^*$$ is the solution of an optimization problem (the decision problem), which requires to solve an argmin/argmax operator which is non-differentiable. To overcome this problem, they proposed to work on a relaxation of the problem. Following this idea is possible to tackle this problem using standard reinforcement learning techniques in which the reward is the negative of the $$argmin$$ argument of the previous equation.

# Set Multicover Problem <a name="section2"></a>
We define here the set multicover problem.  
Let ($$X, \mathcal{S}$$) be a set system, where $X$ is a finite ground set, $$\mathcal{S}$$ is a collection of subsets of $$X$$, and each element $$x \in X$$ has a non-negative demand $$d(x)$$. A set multicover problem requires picking the smallest cardinality sub-collection $$\mathcal{S}'$$ of $$\mathcal{S}$$ such that each point is covered by at least $$d(x)$$ sets from $$\mathcal{S}$$.  
  
Some examples of set multicover problems are pharmacies locations, where you want to place pharmacies to cover all the demands of different locations, and project time scheduling where you want to allocate the available work time of employers to different project that requires certain amount of time to complete.



# Soft Actor Critic <a name="section3"></a>
[Soft actor-critic](https://arxiv.org/pdf/1801.01290.pdf)  is an off-policy actor-critic algorithm based on the maximum entropy framework; the term "soft" is derived from [soft Q-learning](https://arxiv.org/pdf/1702.08165.pdf). In this framework, the optimal policy is the one that optimizes at the same time both the expected cumulative reward value and the entropy of the policy. A parameter $$\alpha$$ is introduced to regulate the importance of the entropy with respect to the reward.

<p align="center"> 
$$
    \pi^* =   \underset\pi{\arg\max} \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}[\gamma^t (r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t))]
$$  
</p>  

Off-policy means that the systems uses data generated from a different policy that the one we are trying to optimize. Indeed, SAC exploits experience which is stored in a replay  buffer and from which batches are randomly sampled for improving the policy. After doing a policy update, the data is kept in the replay buffer leading to a mismatch between the policy that generated the data and the current policy.  
The system is composed of two soft Q-functions, two target soft Q-functions, and an action network. The action network is defined by a Gaussian whose mean and variance are computed by a neural network with parameter $$\phi$$. The soft Q-functions are defined by two neural networks with parameters $$\theta_1$$ and $$\theta_2$$, while the target networks are parametrized with an exponential moving average of the parameters of the soft Q-functions as follow.  

$$
    \bar{\theta_i} \leftarrow (1 - \tau) \bar{\theta_i} + \tau \theta_i
$$  

The presence of two soft Q-function helps to mitigate the effect of positive bias in the policy improvement step and moreover it helps to speed up the training. However only the minimum between the two soft Q-value is used for computing the gradients. The soft Q-function networks are trained to minimize the following equation..

$$
    J_Q(\theta) = \mathbb{E}_{(s_t, a_t) \sim \mathcal{D}}\left[ \frac{1}{2}\left(Q_{\theta}(s_t, a_t) - \left(r(s_t, a_t) + \gamma \ \mathbb{E}_{s_{t+1} \sim p}[V_{\bar{\theta}}(s_{t+1})]\right)\right)^2\right]
$$  

In the first version of this algorithm was present a value-function approximator network which was then abandoned, since the value function is parametrized trough soft Q-function parameters using the following equation.  

$$
    V(s_t) = \mathbb{E}_{a_t \sim \pi} \left[Q(s_t, a_t) - \alpha \ log \ \pi(a_t|s_t)\right]
$$  

Policy network is trained minimizing the following equation.     

$$
J_{\pi}(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[
\ \mathbb{E}_{a_t \sim \pi_\phi} [\alpha \ log \ (\pi \phi(a_t|s_t)) - Q_\theta(s_t, a_t)]\right]
$$

For helping the optimization phase, made by backpropagation, a [reparametrization trick](https://arxiv.org/pdf/1312.6114.pdf) is applied. 
Choosing the optimal temperature $$\alpha$$ parameter is non-trivial and it need to be tuned for each task.  
Reward and entropy can vary a lot for each task and so also the temperature parameter that establish the relative importance of the two. Moreover entropy and rewards vary during training while the policy improve. In \cite{sac2}, they proposed an automated process for tuning this parameter formulating a maximum entropy reinforcement
learning objective, where the entropy is treated as a constraint.  
<p align="center"> 
$$
\underset{\pi 0:T}{\max} \ \mathbb{E}_{\rho_\pi} \left[ \sum_{t=0}^T \gamma^t  r(s_t, a_t) \right]
$$   
$$
s.t. \ \mathbb{E}_{(s_t,a_t) \sim \rho_\pi}
\left[- log(\pi_t(a_t|s_t))\right] \geq \mathcal{H},  \ \forall t
$$  
</p>

where $$\mathcal{H}$$ is the desired minimum expected entropy.  
Exploiting then the duality theorem, it's possible to formulate the following equation for obtaining the optimal $$\alpha$$.  
  

$$
\alpha_t^* = \underset{\alpha_t}{arg min} \mathbb{E}_{a_t \sim \pi^*_t} \left[ - \alpha_t log \pi^*_t(a_t|s_t, \alpha_t) - \alpha_t \bar{\mathcal{H}}\right]
$$ 
  


where $$\bar{\mathcal{H}}$$ is the target entropy.

# Prioritized Experience Replay <a name="section4"></a>
[Schaul et al.](https://arxiv.org/abs/1511.05952) proposed a new method for sampling the experience from the buffer. They suggested sampling with a higher probability the transitions from which is possible to learn more. A possible approximator of this value is the temporal difference error which in some reinforcement learning algorithms is already computed for updating the network parameters. However this greedy technique presents some issues; for example, it requires to sequentially pass all the replay buffer. To avoid expensive sweeps over the entire replay memory, TD errors are only updated for the transitions that are replayed. One consequence is that transitions that have a low TD error on the first visit may not be replayed for a long time (which means effectively never with a sliding window replay memory). Further, it is sensitive to noise spikes (e.g. when rewards are stochastic), which can be exacerbated by bootstrapping.  
To overcome this problem, they proposed sampling proportionally to the priority:

$$
    P(i) = \frac{p_i^{\alpha}}{\sum_k p_i^{\alpha}} 
$$  

where $$p_i = |\delta| + \epsilon$$ with $$\delta$$ being TD error and $$\epsilon$$ a small constant to avoid never revisiting states whose error was zero.  
The prioritized experience introduces bias which can be corrected using importance-sampling weights. These weights are then used for rescaling the gradient during the training. For stability reasons, the weights are normalized with respect to the maximum weight.  
The value $$\beta$$ is increased till it reaches $$1$$ toward the end.    

$$
    w_i = \left( \frac{1}{N} \cdot \frac{1}{P_i}\right)^{\beta}
$$
  

# Experiments
We started our analysis with a quick comparison between SAC and two on-policy algorithms, [PPO](https://arxiv.org/pdf/1707.06347.pdf) and VPG. This round of experiments was conducted using the same backbone for all the algorithms, a 2 fully-connected layers network with $$256$$ neurons for each layer.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/algorithm_comparison.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>
   
From this first trial we noticed that on-policy algorithms converge faster but they are not capable of achieving the same reward value, so we decided to focus our attention on SAC.  
In our successive experiments, we tried to improve the reward and quicken the convergence.  
First of all we launched a grid search for identifying the best combination of hidden dimension and batch size, the batch size was chosen in the range $$[128, 256, 512, 1024]$$, while the hidden dimension was in the range $$[128, 256, 512]$$.  
From the first plot, we can notice that increasing the number of neurons for the layer helps the convergence, instead there isn't a clear relationship between convergence and batch size since in some cases smaller batches converge faster. However, the best combination is the one with $$512$$ as hidden dimension and $$1024$$ as batch size. The reward evaluation plot is more difficult to read since the convergence at evaluation time is less stable than the one at training time.  
<br/><br/>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/train_sac_comparison.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Comparison of train reward during the grid search.
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/eval_sac_comparison.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Comparison of evaluation reward during the grid search.
</div>

<br/><br/>

Once the range of possibilities was reduced, we experimented using the prioritized experience replay. We proposed two little variations to the standard technique, instead of using a liner annealing for the $$\beta$$ value, we used an exponential one, and instead of assigning the maximum priority to the new samples, we noticed that assigning as priority the mean value of the priorities of the last sampled batch was more robust and improved the performances.  
After having fixed the batch size, the hidden dimension, and the type of buffer we launched a random search to find the best learning rate for both policy network and critic networks and the hyper-parameters related to the prioritized experience replay.  
The best configuration we found is presented in the table. This optimized version not only improved the convergence speed but also the reward value.
From the plot, we can notice that the SAC curve start after $$10000$$ steps, this is due to the fact that the replay buffer requires a certain number of pre-collected samples before starting the training. We tried to reduce this number but decreasing it showed a drop in performances.  
Even if on-policy algorithm seems to converge first, they improve little by little during the whole training so they achieve their best evaluation reward after SAC whose evaluation reward is less stable but whose best results are faster.
  
<br/><br/>
    
|----------------------+-----------|     
| **Hyper-parameters** | **Value** |   
| --- | --- |  
| Hidden dimension | 512 |  
| Batch size | 1024 |  
| Policy lr | $$3*10^{-3}$$ |  
| Critic lr | $$3*10^{-2}$$ | 
| Buffer $$\alpha$$ | 0.6 |  
| Starting $$\beta$$ | 0.4 |    
| Annealing rate $$\beta$$ | $$3*10^{-3}$$ |      
  
  <br/><br/>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/final_sac_comparison.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Comparison evaluation reward between the baseline, the best model without PER and the best with it.
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/final_comparison.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Comparison of evaluation reward between PPO, VPG and the best SAC.
</div>
<br/><br/>
# Conclusions <a name="section6"></a>

From our project, we can conclude that the decision focused learning framework it's a promising direction for solving decision making problem.  
Regarding the analyzed algorithms, it seems that on-policy algorithms converge faster and it's particularly important in this context since faster convergence means being able to provide faster solutions to the decision problem. On the other hand SAC delivers better performances reducing the cost of the decision problem.  
After the introduction on the prioritized experience replay technique, the convergence of SAC speeds up, proving the technique to be effective also in this context.  
Moreover we have to take in account that the first $$10000$$ steps, required for filling the buffer, are just explorative steps and they are quicker to execute than the latter ones which also include the policy update phase.   In the end, we can affirm to be satisfied by our project since we achieved better reward, while the increased amount of steps required for the solution seems to be acceptable with respect to the starting phase.  
A possible future improvements for this project would be the implementation of [Prioritized Sequence Experience Replay](https://arxiv.org/pdf/1905.12726.pdf).


