"""Stores code for all the agents.
"""
import numpy as np
import pprint
from collections import defaultdict
import torch
from collections import deque
from collections.abc import Iterable
class UpdateWhileNotTrainingException(Exception):
    pass
class Agent():
    """Base class for all agents. A new agent should implement all functions"""
    def getAction(state, actions):
        raise NotImplemented()
    def update(self,s,a,r,ns,*args, **kwargs):
        raise NotImplemented()
    def train(self,*args,**kwargs):
        pass
    def evaluate(self,*args,**kwargs):
        pass
    def save(self, path):
        pass

class RandomAgent(Agent):
    def getAction(self, state, actions):
        return np.random.choice(actions)
    def update(self,s,a,r,ns,*args, **kwargs):
        pass

class QLearningAgent(Agent):
    def __init__(self):
        self.reset()
    def reset(self):
        self.episodes = 0
        self.gamma = 1
        self.alpha = 0.1
        self.epsilon = 0.1
        self.training = False
        self.totalRewardsEp = 0
        self.exploredEp = 0
        self.actionsEp = 0
    def train(self, epsilon=0.1, 
        gamma=0.8,
        alpha=0.1):
        self.training = True
        self.epsilon = epsilon
        self.episodes += 1
        self.totalRewardsEp = 0
        self.exploredEp = 0
        self.actionsEp = 0
        self.gamma = gamma
        self.alpha = alpha
    def evaluate(self):
        self.training = False
        self.epsilon = 0
    def getAction(self, state, actions):
        '''Get action based on policy. Use exploration while training
        '''
        self.actionsEp += 1
        # actions = [0,2]
        explore = flipCoin(self.epsilon)
        if explore:
            self.exploredEp+= 1
            # print('exploring')
            return np.random.choice(actions)
        else:
            return self.getBestAction(state,actions)
    def getBestAction(self,state, actions):
        raise NotImplemented()
    def update(self, state, action, reward, next_state, next_state_actions=None, finished = False):
        self.totalRewardsEp += reward
        if self.training:
            self._update(state, action, reward, next_state, next_state_actions, finished)
    def _update(self, state, action, reward, next_state, next_state_actions):
        raise NotImplemented()
    def getStats(self):
        stat = {'actions' : self.actionsEp,
            'explored' : self.exploredEp,
            'total rewards' : self.totalRewardsEp
        }
        return stat
    def finishEpisode(self):
        stat = self.getStats()
        # print('Episode Stats = ',stat)
        self.training = False
def flipCoin(p):
    '''flip a coin with prob p of returning True
    '''
    r = np.random.random()
    return r < p

class QTableLearningAgent(QLearningAgent):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        # Q table
        self.Q = defaultdict(lambda: np.zeros(num_actions))
    def getFeatures(self, state, action):
        return state
    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        return self.Q[features][action]
    def getBestAction(self, state, actions):
        '''Given a list of actions, return the best action as per current policy.
        '''
        qValues = [self.getQValue(state, action)
            for action in actions
        ]
        return actions[np.argmax(qValues)]
    def _update(self, state, action, reward, next_state, next_state_actions):
        if  next_state is None:
            next_state_maxQ = 0
        else:
            if next_state_actions is None:
                next_state_actions = range(self.num_actions)
            next_state_maxQ = self.getQValue(next_state, action)
        sampleQ = reward + self.gamma * next_state_maxQ
        qval = self.getQValue(state, action)
        qval = (1- self.alpha) * qval + self.alpha * sampleQ
        features = self.getFeatures(state, action)
        self.Q[features][action] = qval
    def __str__(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.Q)
        return super().__str__()

class ApproximateQLearningAgent(QLearningAgent):
    def __init__(self, num_features, num_actions):
        super().__init__()
        self.num_features = num_features
        self.num_actions = num_actions
        #add 1 for bias. Remember to add a bias feature to the state
        self.weights = np.zeros(shape = (num_actions, num_features + 1 ))#+ 1
        self.qValSumEp = 0
        self.minQ = -200
    def train(self, epsilon=0.1,  *args, **kwargs):
        super().train(epsilon, *args, **kwargs)
        self.qValSum = 0
    def getFeatures(self, state, action):
        '''transforms the input features. 
            Example: add a bias feature
        '''
        return np.append(state,1)
    def getWeights(self, action):
        return self.weights[action]
    def getBestAction(self, state, actions):
        '''Given a list of actions, return the best action as per current policy.
        '''
        qValues = [self.getQValue(state, action)
            for action in actions
        ]
        # print(actions, qValues)
        self.qValSumEp +=np.max(qValues)
        return actions[np.argmax(qValues)]
    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        weights = self.getWeights(action)
        return np.dot(features, weights)

    def _update(self, state, action, reward, next_state, next_state_actions = None, finished = False):
        '''Update Q based on the actual reward vs estimated reward.
        Remember - we don't have any right or wrong labels
        '''
        # if action == 2:
        #     reward += 0.3
        if  next_state is None:
            next_state_maxQ = 0
        else:
            if next_state_actions is None:
                next_state_actions = range(self.num_actions)
            next_state_maxQ = self.getQValue(next_state, 
                self.getBestAction(next_state, next_state_actions))

        rewardBasedQ = reward + self.gamma * next_state_maxQ
        estimatedQ = self.getQValue(state, action)
        diff = rewardBasedQ - estimatedQ
        # print(diff)
        features = self.getFeatures(state,action)
        weights = self.getWeights(action)
        weights += self.alpha * diff * features
        self.weights[action] = np.clip(weights, -10, 10)
        # print(self.weights)
    def getStats(self):
        stat = super().getStats()
        stat['averageQVal'] = self.qValSumEp/self.actionsEp
        return stat
    def __str__(self):
        return str(self.weights)


class HeuristicApproxQLearner(ApproximateQLearningAgent):
    def __init__(self):
        super().__init__(4,4)

    def getFeatures(self, s, action):
        angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        hover_targ = 0.55 * np.abs(
            s[0]
        )  # target y should be proportional to horizontal offset

        angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
        hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

        if s[6] or s[7]:  # legs have contact
            angle_todo = 0
            hover_todo = (
                -(s[3]) * 0.5
            )  # override to reduce fall speed, that's all we need after contact
        features = np.zeros(4)
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            features[2] = 1
        elif angle_todo < -0.05:
            features[3] = 1
        elif angle_todo > +0.05:
            features[1] = 1

        # As per heuristic
        # a = 0
        # if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        #     a = 2
        # elif angle_todo < -0.05:
        #     a = 3
        # elif angle_todo > +0.05:
        #     a = 1
        transformed_state = features
        return super().getFeatures(transformed_state, action)

class LanderQTableAgent(QTableLearningAgent):
    def __init__(self, num_actions):
        super().__init__(num_actions)
    def getFeatures(self, state, action):
        s = state
        #discretization
        state = (min(5, max(-5, int((s[0]) / 0.05))), \
            min(5, max(-1, int((s[1]) / 0.1))), \
            min(3, max(-3, int((s[2]) / 0.1))), \
            min(3, max(-3, int((s[3]) / 0.1))), \
            min(3, max(-3, int((s[4]) / 0.1))), \
            min(3, max(-3, int((s[5]) / 0.1))), \
            int(s[6]), \
            int(s[7]))
        return state

class DQNAgentModel(torch.nn.Module):
    def __init__(self, ip_size, op_size, lr = 0.1, gamma = 0.95):
        super().__init__()
        self.ip_size = ip_size
        self.op_size = op_size
        hidden_size = 64
        self.net = torch.nn.Sequential(
                torch.nn.Linear(ip_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, op_size)
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net.to(self.device)
        self.maxmemory = 1000000
        self.memory = deque(maxlen=self.maxmemory)
        self.batch_size = 32
        self.lr = lr
        self.gamma = gamma
        self.optim = torch.optim.SGD(lr = lr, params = self.parameters())
        self.criterion = torch.nn.MSELoss()
        self.losses = []
        self.steps = 0
    def forward(self, x):
        if not isinstance(x, torch.Tensor):# assumes bs 1
            x = torch.tensor(np.array(x), dtype = torch.float)
        return self.net(x)
    def get_random_memories(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        finished = []
        for i in indices:
            states.append(self.memory[i][0])
            actions.append(self.memory[i][1])
            rewards.append(self.memory[i][2])
            next_states.append(self.memory[i][3])
            finished.append(int(self.memory[i][4]))
        return {'states' :torch.tensor(np.array(states), dtype = torch.float).reshape(len(states),self.ip_size),
                'actions' : torch.tensor(actions, dtype = torch.long),
                'rewards': torch.tensor(rewards, dtype = torch.float),
                'next_states' : torch.tensor(np.array(next_states), dtype = torch.float).reshape(len(states),self.ip_size),
                'finished' : torch.tensor(finished, dtype = torch.int)
               }
    def addObservation(self,state, action, reward, next_state, finished):
        self.memory.append([state, action, reward, next_state, finished])
    def train_step(self, states, labels):
        # pdb.set_trace()
        states = states.to(self.device)
        labels = labels.to(self.device)
        self.steps += 1
        self.optim.zero_grad()
        pred = self.forward(states).squeeze()
#         print(pred, labels)
        loss = self.criterion(labels, pred)
        loss.backward()
        self.losses.append(loss.item())
        # if self.steps % 50 == 0:
        #     print(f'{self.steps}: loss = {loss}')#, model = {self.net[0].weight.data.numpy()}')
        self.optim.step()
    def generate_label(self, memories):
        states = memories['states']
        actions = memories['actions']
        labels = self.forward(states)
        next_state_qvals = torch.max(self.forward(memories['next_states']), axis = 1)[0] * (1-memories['finished'])
        
        labels[range(labels.shape[0]), actions] = \
            memories['rewards'] + self.gamma * next_state_qvals
        return labels
    def replay_experiences(self):
        if len(self.memory)<self.batch_size:
            return
        memories = self.get_random_memories(self.batch_size)
        states = memories['states']
        labels = self.generate_label(memories)
#         print(states, labels)
        self.train_step(states, labels)
class DeepQLearningAgent(QLearningAgent):
    def __init__(self, state_dim, action_dim, lr = 0.01, gamma = 0.95):
        super().__init__()
        self.model = DQNAgentModel(state_dim, action_dim, lr, gamma)
        self.num_updates = 0
    def train(self,*args,**kwargs):#alpha and gamma not used. fix later
        super().train(*args,**kwargs)
        self.model.train()
    def getBestAction(self,state, actions):
        if not isinstance(state, Iterable):
            state = [state]
        qVals = self.model([state])[0].data
        best_a = max(zip(qVals[actions], actions))[1]
#         print(actions,qVals, best_a)
        return best_a
    def _update(self, state, action, reward, next_state, next_state_actions, finished = False):
        # print(state, action, reward, next_state, finished)
        self.num_updates += 1
        self.model.addObservation(state, action, reward, next_state, finished )
        #high scale - numerical instability
        self.model.replay_experiences()
    def save(self, path):
        torch.save(self.model.state_dict(),path)
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    def __str__(self) -> str:
        return str(self.model.state_dict()['net.0.weight'][0])
