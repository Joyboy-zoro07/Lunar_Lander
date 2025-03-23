import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
def train_1ep(env, agent, seed = None, render=False, max_steps = 300, verbose = False):
    s = env.reset(seed=seed)
    total_reward = 0
    steps = 0

    action_counts = [0,0,0,0]
    while True:
        a = agent.getAction(s, range(4)) #env.action_space.shape
        action_counts[a] += 1
        
        ns, r, done, info = env.step(a)
        total_reward += r
        if render:
            still_open = env.render()
            if still_open == False: break
        if verbose:
            print(f"step {steps} a = {a} s = {s} r= {r}")
        agent.update(s, a, r, ns, None, finished = done)
        s = ns
        steps += 1

        if done or steps >max_steps:
            # print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
            break
    return total_reward, steps, action_counts

# def getEpsilon(iter):

#     threshold = 50
#     if iter > 200:
#         threshold = 10
#     if iter > 2000:
#         threshold = 5
#     if iter > 5000:
#         threshold = 1
#     if iter > 7500:
#         threshold = 0
#     return threshold/100 # TODO: change to threshold

def getEpsilon(iter):
    return max(0.2, np.power(0.996,iter))


def train(env, agent, seed=None, render=False, 
        episodes = 1, name = 'UnnamedAgent', verbose = False, max_steps = 1000):
    reward_allep = []
    steps_allep = []

    avg_rewards = []
    avg_steps = []

    for ep in range(1,episodes+1):
        s = env.reset(seed=seed)
        agent.train(
            epsilon=getEpsilon(ep),
            gamma=0.95,
            alpha=0.001
        )
        total_reward, steps, action_counts = train_1ep(env,
                                                seed = 0,# np.random.randint(0,100),
                                                render=render,
                                                agent = agent,
                                                max_steps = max_steps,
                                                verbose = verbose)
        reward_allep.append(total_reward)
        steps_allep.append(steps)
        if ep%50 == 0:
            render = True
        if ep%51 == 0:
            render = False
        if ep%50 == 0:
            avg_rewards.append(np.mean(reward_allep))
            avg_steps.append(int(np.mean(steps_allep)))
            print(f"ep {ep} steps {avg_steps[-1]} reward {avg_rewards[-1]:+0.2f}, action_counts = {action_counts}, ep = {getEpsilon(ep)}")
            reward_allep = []
            steps_allep = []
            # print(agent)
            # agent.save(f'./checkpoints/{name}_{ep}')
    fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
    fig.suptitle(name, fontsize=16)
    plt.grid(True)
    ax1.plot( avg_rewards, label= name)
    ax1.set_title('Average rewards')
    ax2.plot( avg_steps, label=name)
    ax2.set_title('Average steps')
    np.savetxt(f'./results/{name}_avg_rewards.txt', np.array(avg_rewards))
    np.savetxt(f'./results/{name}_avg_steps.txt', np.array(avg_steps))
    plt.savefig(f"results/{name}.png")

def evaluate(env, agent, episodes, max_steps = 1000):
    env.reset()
    for ep in range(episodes):
        agent.evaluate()
        total_reward, steps, action_counts = train_1ep(env,
                                                seed = 0, #np.random.randint(0,100), # pick it randomly 
                                                render=True,
                                                agent = agent,
                                                max_steps = max_steps,
                                                verbose = False)
        print(f"ep {ep} steps {steps} reward {total_reward:+0.2f}, action_counts = {action_counts}")

import agents
import lunar_lander

def main():

    env = lunar_lander.LunarLander()
    # agent = agents.RandomAgent()
    # agent = agents.ApproximateQLearningAgent(8, 4)
    # agent = agents.HeuristicApproxQLearner()
    # agent = agents.LanderQTableAgent(4)
    agent = agents.DeepQLearningAgent(8,4, lr =0.001, gamma = 0.99)
    name = type(agent).__name__
    agent.load(f'saved_results/DeepQLearningAgent_1000_3')
    print(agent)
    episodes= 1000
    # uncomment the following to enable training.

    # train(env, 
    #     agent, 
    #     seed = 0, 
    #     render = False, 
    #     episodes= episodes, 
    #     max_steps = 1000,
    #     name = name,
    #     verbose = False)

    plt.show()
    print('evaluating')
    evaluate(env, agent, episodes=1, max_steps = 3000)
    print(agent)

if __name__ == '__main__':
    main()
