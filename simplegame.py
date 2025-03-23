import numpy as np
import agents as agents
class Environment:
    def __init__(self):
        self.reset()
    def reset(self):
        self.state = self.getInitialState()
    def isGoal(self,state):
        raise NotImplemented()
    def getInitialState(self):
        raise NotImplemented()
    def getPossibleActions(self):
        raise NotImplemented()
    def renderEnv(self):
        raise NotImplemented()
    def step(self, action):
        '''Returns reward and next state
        '''
        raise NotImplemented()
class OneDtarget(Environment):
    def __init__(self, targetPoint=5):
        self.targetPoint = targetPoint
        self.minPoint = 0
        self.maxPoint = 10
        super().__init__()
    def isGoal(self,state):
        return state == self.targetPoint
    def getInitialState(self):
        return np.random.randint(self.minPoint, self.maxPoint+1)
    def getPossibleActions(self):
        if self.state == self.minPoint:
            return [1]
        elif self.state == self.maxPoint:
            return [0]
        else:
            return [0,1]
    def renderEnv(self):
        arr = np.zeros(self.maxPoint+1, dtype = int)
        arr[self.state] = 1
        arr[self.targetPoint] = 100
        print(arr)
    def step(self,  action):
        '''Returns reward and next state
        '''
        if action not in self.getPossibleActions():
            raise Exception(f'Incorrect action {action} for state {self.state}')
        
        next_state = self.state + (action*2)-1
        reward = 0
        if self.isGoal(next_state):
            reward = 1
        else:
            #intermittent rewards
            closer = abs(self.state-self.targetPoint) - abs(next_state-self.targetPoint)
            # reward  = closer/(self.maxPoint-self.minPoint)
        self.state = next_state
        return reward, self.state
def playOneEpisode(
        env, 
        agent,
        maxSteps,
        update = True,
        render = False):
    steps = 0
    rewards = 0
    env.reset()
    while steps < maxSteps:
        done = env.isGoal(env.state)
        if done:
            # print('Reached goal in {} steps. Total Reward = {}'.format(steps, rewards))
            break
        steps += 1
        initial_state = env.state
        possible_actions = env.getPossibleActions()
        action = agent.getAction(env.state, possible_actions)
        #print(f'step {steps}: Possible actions = {possible_actions}, action predicted = {action}')
        reward, next_state = env.step(action)
        
        if render:
            env.renderEnv()
        done = env.isGoal(env.state)
        if update:
            next_state_actions = env.getPossibleActions()
            agent.update(initial_state, action, reward, next_state, next_state_actions, finished = done)
        rewards+=reward
        if done:
            # print('Reached goal in {} steps. Total Reward = {}'.format(steps, rewards))
            break
        
    return rewards, steps
        
if __name__ == '__main__':
    agent = agents.ApproximateQLearningAgent(1,2)
    env = OneDtarget()
    episodes = 1000
    maxStep = 50
    for ep in range(episodes):
        print(f'episode {ep+1}')
        agent.train(epsilon = 0.2, gamma=0.9, alpha = 0.001)
        playOneEpisode(env, agent, maxStep)
        # print(agent.weights)
    print('training finished')
    agent.evaluate()

    for _ in range(5):
        playOneEpisode(env, agent, maxStep, render = True)
    print(agent.weights, agent.epsilon)
    for i in range(env.minPoint, env.maxPoint+1):
        print(f'position {i}: action {agent.getBestAction(i, [0,1])}')
    # for _ in range(1):
    #     playOneEpisode(env, agent, maxStep, update = False)
    # print(agent.weights)
    
    



