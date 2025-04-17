import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

# 相当于使用单GPU训练算法。不再多言。


################################## PPO Policy ##################################
''' 存储交互数据 '''
class RolloutBuffer:
    def __init__(self):
        self.actions = [] # 动作空间
        self.states = [] # 状态空间
        self.logprobs = [] # 计算log概率。
        self.rewards = [] # 单步reward
        self.state_values = [] # v值
        self.is_terminals = [] # 终止标志，代表一个循环终止。
    
    '''删除元素，保留对象。减少新建开销。'''
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        # 传入外部参数
        self.has_continuous_action_space = has_continuous_action_space
        
        # 这里在处理连续空间。可以看到我们在初始化。
        # 初始化的时候矩阵整个都是一样的，初始标准差平方。
        # 注意标准差和探索率相关，所以初始化值有意义。
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        # 这里处理连续动作空间，单纯是用一个简单分布拟合动作空间。具体会在PPO部分里涉及。
        # 这里只要知道我们希望它分布在(-1,1)
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # A网络，一般是用于PU，也就是策略更新。
        # 在这里就是一个函数拟合器（神经网络），用于表示策略函数
        # 同时它还实现了DPG的思路，在连续动作空间下将会使用确定性策略。

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        # C网络。一般是用于PE，也就是策略评估。
        # 仍然是一个函数拟合器，用于表示值函数。
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
        # 在子类实现
    
    # 发生在rollout阶段，主要是采集数据。首先随机采样一个动作，并同时计算对数概率和v（主要是为下一步做准备）旧策略产生数据罢了。
    def act(self, state):
        # 连续动作空间处理：生成一个多元正态分布，用于拟合动作（或者也可以说，某个策略）空间
        # * 使用神经网络self.actor处理当前状态，生成动作的均值action_mean
        # * 创建一个协方差矩阵cov_mat，通过将self.action_var（动作方差）转换为矩阵，并增加一个维度。
            # 如果我们只是用一个一元正态分布的话就只需要方差就行了。现在需要用协方差矩阵表示不同动作维度的方差【注意这里是动作维度而不是动作空间，动作空间是连续的】
            # 如果用的是对角矩阵，那就说明数据之间无关。一般复杂度到这个程度就行了。
        # * 使用均值和协方差矩阵创建一个多元正态分布`MultivariateNormal`。具体可以看数学（
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)

        # 非连续动作空间
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            # 用actor网络计算各个动作均值，然后用categorical创建分类分布

        # 这里是Critic过程。
        action = dist.sample() # 采样一个动作。
        action_logprob = dist.log_prob(action) # 做对数概率。
        state_val = self.critic(state) # 直接运行critic网络，计算价值。

        return action.detach(), action_logprob.detach(), state_val.detach()
        # 返回数据，并把他从计算图中分离。也就是说这些数据目前【被存储】而不拿去做参数更新。
        '''detach的用法！'''
    
    # 发生在训练阶段，
    def evaluate(self, state, action):
        # 这是连续空间，和上面差不多。
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        # 现在是离散空间
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        # 处理连续动作空间。
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        # 分别定义衰减值，Clip值（这是PPOC），epoch数
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # 设置一个Buffer。
        self.buffer = RolloutBuffer()

        # 新策略
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)

        # 用的是Adam优化器
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        # 旧策略
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 定义计算MSE loss
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    # 动作交互并存储到buffer
    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    # 策略更新过程
    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        # 计算累计return！
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
