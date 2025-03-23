## Environment
- There are two environment versions: discrete or continuous.
- The landing pad is always at coordinates (0,0). 
- The coordinates are the first two numbers in the state vector.
- Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

## Action Space

There are four discrete actions available: 

1. do nothing 0
2. fire left orientation engine 1 
3. fire main engine 2 
4. fire right orientation engine. 3

For continuous, Action is two floats [main engine, left-right engines].
    1. Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
    2. Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

## State Space
Total 8 dimensions
1. coordinates of the lander in `x` & `y`
2. its linear velocities in `x` & `y`
3. angle
4. angular velocity
5. two booleans that represent whether each leg is in contact with the ground or not

The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

```
s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0
```

## Rewards
1. Reward for moving from the top of the screen to the landing pad and coming to rest is about 100-140 points.
2. If the lander crashes, it receives an additional -100 points.
3. If it comes to rest, it receives an additional +100 points.
4. Each leg with ground contact is +10 points.
5. Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
6. Solved is 200 points.
## Objective

Create an agent which achieves maximum reward.

## Algorithm

## MDP

Issues:
1. Environment is too complex to understand
2. States like angular speed, velocity etc are difficult to understand and make heuristics

Understanding:
1. Heuristic is probably trying to move the lander towards the angle it makes with center.
2. There is probably no uncertainity.

Approach:
- [x] Start with a random agent to see how it behaves.
- [ ] MDP?
  - MDP will not be suitable as it will take ages to come up with the MDP model parameters i.e. the transition probs.
- [ ] Q Learning
  - https://courses.cs.washington.edu/courses/csep573/22wi/project3/index.html
  - This can be suitable for the discrete actions.
  - We can learn from experience and exploit actinos which have high chances of success while exploring slightly.
  - Problems
    -  the state space is pretty large (find the exact value) and it might again take more time to learn properly.
    -  We need to discretize the state space. This needs knowledge of max and minimum values. This will require deep understanding of state feature ranges and distributions.
  - Try changing the parameters, 
    - discount $\gamma$
    - epsilon $\epsilon$
    - learning rage $\alpha$
- [ ] Approximate Q Learning
  - [ ] This will be more suitable than Q learning as the state variables are continuous.
  - [ ] Will help with generalization
  - [ ] Prolem : need to find out features for $Q(s,a)$
    - [ ] may be i can have 4 functions for 4 separate actions. Use perceptron type logic to find the highest value out of them
    - [ ] https://www.cs.swarthmore.edu/~bryce/cs63/s16/slides/3-25_approximate_Q-learning.pdf
- [ ] Deep Q learning
  - [ ] Better than approx Q learning

# Results
## Random
ep 50 steps 65 reward -119.61, action_counts = [17, 9, 20, 16], ep = 0.8184024506760997
ep 100 steps 84 reward -95.92, action_counts = [14, 17, 15, 12], ep = 0.6697825712726458
ep 150 steps 82 reward -110.47, action_counts = [15, 16, 14, 14], ep = 0.5481516977496729
ep 200 steps 84 reward -107.75, action_counts = [13, 14, 17, 18], ep = 0.44860869278059695
ep 250 steps 67 reward -144.03, action_counts = [17, 15, 29, 18], ep = 0.3671424535662421
ep 300 steps 65 reward -103.25, action_counts = [17, 17, 13, 14], ep = 0.3004702837458487
ep 350 steps 65 reward -106.58, action_counts = [20, 16, 14, 11], ep = 0.24590561657294563
ep 400 steps 65 reward -111.04, action_counts = [9, 19, 14, 17], ep = 0.20124975923831603
ep 450 steps 84 reward -93.46, action_counts = [18, 16, 22, 17], ep = 0.2
ep 500 steps 65 reward -105.17, action_counts = [22, 21, 13, 17], ep = 0.2
ep 550 steps 63 reward -116.05, action_counts = [17, 16, 20, 14], ep = 0.2
ep 600 steps 64 reward -102.11, action_counts = [14, 21, 14, 14], ep = 0.2
ep 650 steps 83 reward -123.01, action_counts = [19, 15, 16, 21], ep = 0.2
ep 700 steps 64 reward -85.91, action_counts = [15, 18, 19, 11], ep = 0.2
ep 750 steps 65 reward -98.11, action_counts = [17, 21, 15, 20], ep = 0.2
ep 800 steps 65 reward -114.19, action_counts = [14, 16, 20, 17], ep = 0.2
ep 850 steps 63 reward -115.10, action_counts = [17, 13, 6, 20], ep = 0.2
ep 900 steps 67 reward -114.77, action_counts = [9, 25, 13, 13], ep = 0.2
ep 950 steps 67 reward -114.19, action_counts = [17, 17, 13, 16], ep = 0.2
ep 1000 steps 63 reward -117.63, action_counts = [12, 17, 27, 21], ep = 0.2
## Q table

    agent = agents.LanderQTableAgent(4)

Hyper parameters

    train(env, 
        agent, 
        seed = 47, 
        render = False, 
        episodes= 10000, 
        name = name,
        verbose = False)
      agent.train(
            epsilon=getEpsilon(ep),
            gamma=0.95,
            alpha=0.3
        )

  def getEpsilon(iter):

      threshold = 50
      if iter > 200:
          threshold = 10
      if iter > 2000:
          threshold = 5
      if iter > 5000:
          threshold = 1
      if iter > 7500:
          threshold = 0
      return threshold/10
![training rewards](./mycode/saved_results/LanderQTableAgent.png)

Agent was not learning as there was a problem with Q value updates

incorrect 
  
    qval += (1- self.alpha) * qval + self.alpha * sampleQ

correct
  
    qval = (1- self.alpha) * qval + self.alpha * sampleQ

## Appoximate q learning with heuristic based onehot features

```
  agent = agents.HeuristicApproxQLearner()

        features = np.zeros(4)
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            features[2] = 1
        elif angle_todo < -0.05:
            features[3] = 1
        elif angle_todo > +0.05:
            features[1] = 1
```
![training rewards](./mycode/saved_results/HeuristicApproxQLearner.png)
Conclusion:
- Agent does not learn well with high learning rate.
- high exploration generally yields bad rewards.
- low exploration with low learning rate yields +ve rewards
  ```
          agent.train(
            epsilon=getEpsilon(ep),
            gamma=0.95,
            alpha=0.001
        )

        epsilon = 0.1
  ```
- It still does not learn the best heuristics. onehot features are not assigned highest labels. Example: 
  ```
  ep 1400 steps 240 reward +79.16
  [[ 0.         -0.5067327   5.96948817  0.6023203   7.53688717]
  [ 0.          0.55166405  5.37667624  0.14314042  6.50684866]
  [ 0.         -0.14081854  9.76332673  0.70381361  6.93361496]
  [ 0.         -0.02792927  6.12005603  2.08995898  6.90837691]]
  ep 1500 steps 224 reward +158.11
  [[ 0.         -0.80849731  6.20459922  0.65929478  8.89059442]
  [ 0.          1.14078767  5.80344831  0.17726023  7.97594429]
  [ 0.         -0.36009664  9.96610508  0.71622254  8.89177503]
  [ 0.         -0.02867782  6.35537343  3.24155341  8.32965188]]
  ```
-Attains maximum reward faster. But hard to design these features in general
- Proves that code and learning algorithm is correct.
- Agent can learn with better features. Learning rate needs to be slowed down once model converges. May be problem with perceptron.

## Approximate Q Learning with vannilla state features

10000 episodes. Exploration - decaying (same as q table)

  agent = agents.ApproximateQLearningAgent(8, 4)
    
![training rewards](./mycode/saved_results/ApproximateQLearningAgent.png)

Conclusion:
- Learns something but converges at reward  = -100 and avg number of steps also decreases. Looks like it's trying to reduce tiem time in air. 
- Incresing number of exploration steps does not help.
- Underfitting issue. May be linear function is not useful here
- Use deep learning or design non-linear features which are linearly separable.

## Deep Q learning
even deep model with only linear regression was not able to learn proper separation for 1dtarget in simple game.
```
0 tensor(1)
1 tensor(1)
2 tensor(1)
3 tensor(1)
4 tensor(1)
5 tensor(1)
6 tensor(1)
7 tensor(1)
8 tensor(0)
9 tensor(0)
10 tensor(0)
```
This is probably because of the sample imbalanace. The terminal states are not seen much often. Leading to model not being very precise. 
Also adding intermediate rewards if target is closer because of action - didnt help.
This is probably because the task is a regression task. Because of gamma, the prediction decreases exponentially with distance. Probably that's why linear model is not able to model that.

Let's try non linear model with more parameters.
It worked. Though there are some local minimas, the model learns accurate weight most of the times
```
0 tensor(1)
1 tensor(1)
2 tensor(1)
3 tensor(1)
4 tensor(1)
5 tensor(0)
6 tensor(0)
7 tensor(0)
8 tensor(0)
9 tensor(0)
10 tensor(0)
```
```
agent.evaluate()
simplegame.playOneEpisode(env, agent, 100, render = True)
[  0   0   0   0   0 100   0   0   0   1   0]
[  0   0   0   0   0 100   0   0   1   0   0]
[  0   0   0   0   0 100   0   1   0   0   0]
[  0   0   0   0   0 100   1   0   0   0   0]
[  0   0   0   0   0 100   0   0   0   0   0]
```

### lunar lander

  agent = agents.DeepQLearningAgent(8,4, lr =0.001, gamma = 0.99)
  
  epsilon
    return max(0.01, np.power(0.996,iter))


ep 50 steps 68 reward -85.34, action_counts = [18, 22, 16, 10], ep = 0.8184024506760997
ep 100 steps 75 reward -112.05, action_counts = [13, 13, 32, 14], ep = 0.6697825712726458
ep 150 steps 77 reward -75.85, action_counts = [34, 8, 15, 9], ep = 0.5481516977496729
ep 200 steps 107 reward -64.11, action_counts = [16, 23, 49, 14], ep = 0.44860869278059695
ep 250 steps 117 reward -63.44, action_counts = [16, 23, 60, 16], ep = 0.3671424535662421
ep 300 steps 97 reward -45.98, action_counts = [32, 6, 14, 7], ep = 0.3004702837458487
ep 350 steps 93 reward -37.17, action_counts = [19, 18, 63, 8], ep = 0.24590561657294563
ep 400 steps 123 reward -30.94, action_counts = [43, 4, 16, 3], ep = 0.20124975923831603
ep 450 steps 123 reward -19.43, action_counts = [11, 30, 68, 26], ep = 0.1647032961586129
ep 500 steps 162 reward +23.94, action_counts = [28, 29, 91, 28], ep = 0.13479358121064025
ep 550 steps 190 reward +29.85, action_counts = [9, 35, 92, 24], ep = 0.11031539719819584
ep 600 steps 212 reward +17.46, action_counts = [23, 42, 94, 28], ep = 0.09028239141431083
ep 650 steps 212 reward +33.90, action_counts = [31, 25, 91, 32], ep = 0.07388733038637085
ep 700 steps 242 reward -12.06, action_counts = [17, 38, 112, 24], ep = 0.060469572262120554
ep 750 steps 222 reward +29.27, action_counts = [31, 33, 93, 24], ep = 0.04948844613065497
ep 800 steps 317 reward +52.81, action_counts = [14, 41, 117, 48], ep = 0.040501465593480175
ep 850 steps 329 reward -6.19, action_counts = [11, 44, 123, 123], ep = 0.03314649869767791
ep 900 steps 234 reward +3.14, action_counts = [31, 32, 86, 21], ep = 0.027127175765511748
ep 950 steps 194 reward -15.31, action_counts = [27, 37, 87, 21], ep = 0.02220094712641612
ep 1000 steps 197 reward -21.54, action_counts = [18, 36, 93, 23], ep = 0.018169309535589467

evaluating
ep 0 steps 174 reward -16.49, action_counts = [20, 37, 96, 21]
![positive exploration](mycode/saved_results/DeepQLearningAgent.png)
#### Retraining

epsilon =  max(0.01, np.power(0.996,iter)*0.5)
agent.load(f'./checkpoints/{name}_{800}')

started from 800th checkpoint. exploration started from 0.5 reduced by factor of 0.996 every episode
ep 50 steps 104 reward -7.67, action_counts = [23, 23, 39, 36], ep = 0.40920122533804987
ep 100 steps 177 reward +16.80, action_counts = [15, 19, 64, 21], ep = 0.3348912856363229
ep 150 steps 337 reward +50.74, action_counts = [311, 213, 163, 314], ep = 0.27407584887483644
ep 200 steps 378 reward +50.00, action_counts = [18, 39, 136, 43], ep = 0.22430434639029848
ep 250 steps 592 reward +80.29, action_counts = [20, 38, 140, 48], ep = 0.18357122678312104
ep 300 steps 657 reward +101.35, action_counts = [15, 31, 88, 27], ep = 0.15023514187292436
ep 350 steps 717 reward +116.64, action_counts = [504, 135, 161, 201], ep = 0.12295280828647281
ep 400 steps 847 reward +139.36, action_counts = [536, 100, 225, 140], ep = 0.10062487961915802
ep 450 steps 917 reward +139.45, action_counts = [595, 120, 137, 149], ep = 0.08235164807930645
ep 500 steps 885 reward +145.76, action_counts = [349, 132, 213, 307], ep = 0.06739679060532013
tensor([ 0.0706, -0.5410, -0.3318, -0.0121,  0.1886,  0.0762,  0.0182, -0.1058])

evaluating
ep 0 steps 406 reward +234.13, action_counts = [82, 74, 185, 65]
![positive exploration](mycode/saved_results/DeepQLearningAgent_2.png)
- Having higher exploration and diverse set of data is important. when exploration decreases, model overfits to exploited examples as replay samples more and more from latest examples. Otherwise training sufferes as can be seen between 900 to 1000ths episode.
- Seed needs to change otherwise model overfits for that particular seed.
- intelligent replay can result in faster convergence as exploration can be decreased
- episode lenght increases as model learns more. This affects replay memory and fills it up with data from similar states. Having high exploration helps avoid these scenarios.
- Let's test by capping exploration to 0.2

### 0.2 max epsilon

return max(0.2, np.power(0.996,iter))
```
tensor([-0.2616,  0.3166, -0.0272, -0.2830, -0.2549, -0.1810, -0.0249,  0.2511])
ep 50 steps 68 reward -111.72, action_counts = [9, 25, 16, 13], ep = 0.8184024506760997
ep 100 steps 77 reward -147.07, action_counts = [16, 13, 42, 12], ep = 0.6697825712726458
ep 150 steps 76 reward -152.62, action_counts = [7, 25, 12, 14], ep = 0.5481516977496729
ep 200 steps 78 reward -139.31, action_counts = [9, 16, 27, 19], ep = 0.44860869278059695
ep 250 steps 82 reward -94.38, action_counts = [16, 17, 44, 15], ep = 0.3671424535662421
ep 300 steps 96 reward -109.75, action_counts = [17, 16, 43, 12], ep = 0.3004702837458487
ep 350 steps 103 reward -161.28, action_counts = [16, 21, 45, 21], ep = 0.24590561657294563
ep 400 steps 95 reward -52.49, action_counts = [15, 17, 37, 27], ep = 0.20124975923831603
ep 450 steps 102 reward -29.81, action_counts = [22, 21, 55, 19], ep = 0.2
ep 500 steps 125 reward -9.84, action_counts = [25, 21, 72, 24], ep = 0.2
ep 550 steps 150 reward +4.03, action_counts = [38, 13, 82, 24], ep = 0.2
ep 600 steps 151 reward +20.05, action_counts = [44, 16, 89, 22], ep = 0.2
ep 650 steps 160 reward +24.61, action_counts = [21, 21, 80, 25], ep = 0.2
ep 700 steps 187 reward +30.83, action_counts = [21, 21, 83, 24], ep = 0.2
ep 750 steps 215 reward +43.11, action_counts = [20, 13, 67, 21], ep = 0.2
ep 800 steps 272 reward +47.11, action_counts = [32, 20, 95, 26], ep = 0.2
ep 850 steps 385 reward +71.00, action_counts = [47, 25, 119, 22], ep = 0.2
ep 900 steps 557 reward +80.81, action_counts = [59, 39, 182, 38], ep = 0.2
ep 950 steps 591 reward +90.95, action_counts = [429, 122, 188, 262], ep = 0.2
ep 1000 steps 638 reward +94.05, action_counts = [40, 34, 154, 53], ep = 0.2
tensor([-0.0424,  0.6711,  0.1539, -0.2784, -0.2472, -0.6159, -0.0485,  0.2555])
```
evaluating
ep 0 steps 328 reward +272.64, action_counts = [14, 83, 110, 121]
![positive exploration](mycode/saved_results/DeepQLearningAgent_3.png)
- intuition was correct. achieved maximum reward. For reply to work efficiently, data should be diverse. Having a positive clipping for exploration helps


Random seed for evaluation
evaluating
ep 0 steps 289 reward +44.85, action_counts = [21, 65, 153, 50]
evaluating
ep 0 steps 89 reward -114.99, action_counts = [0, 26, 23, 40]
### random seed
```
tensor([-0.2175,  0.3233,  0.1665, -0.0903,  0.1521, -0.0004,  0.3197,  0.0137])
ep 50 steps 102 reward -210.35, action_counts = [11, 19, 42, 17], ep = 0.8184024506760997
ep 100 steps 113 reward -136.15, action_counts = [19, 18, 25, 26], ep = 0.6697825712726458
ep 150 steps 121 reward -61.55, action_counts = [9, 20, 30, 13], ep = 0.5481516977496729
ep 200 steps 116 reward -41.35, action_counts = [18, 15, 54, 12], ep = 0.44860869278059695
ep 250 steps 144 reward -26.89, action_counts = [45, 28, 101, 32], ep = 0.3671424535662421
ep 300 steps 228 reward -0.19, action_counts = [20, 28, 98, 24], ep = 0.3004702837458487
ep 350 steps 468 reward +23.17, action_counts = [44, 82, 235, 75], ep = 0.24590561657294563
ep 400 steps 484 reward +11.25, action_counts = [17, 44, 109, 52], ep = 0.20124975923831603
ep 450 steps 392 reward -5.13, action_counts = [24, 42, 114, 49], ep = 0.2
ep 500 steps 340 reward -41.26, action_counts = [24, 42, 110, 57], ep = 0.2
ep 550 steps 420 reward -8.99, action_counts = [18, 70, 147, 61], ep = 0.2
ep 600 steps 376 reward -40.92, action_counts = [16, 82, 203, 79], ep = 0.2
ep 650 steps 431 reward -26.49, action_counts = [36, 51, 163, 68], ep = 0.2
ep 700 steps 476 reward -19.30, action_counts = [10, 30, 109, 51], ep = 0.2
ep 750 steps 587 reward -12.15, action_counts = [57, 85, 244, 60], ep = 0.2
ep 800 steps 649 reward -9.57, action_counts = [131, 84, 383, 77], ep = 0.2
ep 850 steps 688 reward -18.99, action_counts = [22, 66, 168, 48], ep = 0.2
ep 900 steps 697 reward -2.09, action_counts = [11, 51, 119, 51], ep = 0.2
ep 950 steps 612 reward -18.21, action_counts = [16, 31, 114, 47], ep = 0.2
ep 1000 steps 707 reward -8.41, action_counts = [113, 85, 355, 95], ep = 0.2
tensor([-0.1996,  0.9746,  0.0705, -0.7477,  0.7451,  0.2523,  0.5796, -0.1189])
evaluating
ep 0 steps 1001 reward -116.12, action_counts = [324, 51, 580, 46]
tensor([-0.1996,  0.9746,  0.0705, -0.7477,  0.7451,  0.2523,  0.5796, -0.1189])
```
![random seed](mycode/saved_results/DeepQLearningAgent_4.png)