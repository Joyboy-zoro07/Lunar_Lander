
# 🚀 Lunar Lander – Reinforcement Learning Project

This project implements and compares multiple **Reinforcement Learning (RL) algorithms** to solve the **Lunar Lander problem**, a classic physics-based control task.  
The goal is to train an autonomous agent that can **safely and efficiently land a spacecraft** using trial-and-error learning.

---

## 📌 Problem Statement

Autonomous landing is a **sequential decision-making problem** where:
- The environment is unstable and noisy
- Rewards are delayed
- No labeled “correct actions” exist
- Decisions affect long-term outcomes

Traditional rule-based or supervised learning approaches fail here.  
This project demonstrates how **Reinforcement Learning** can learn optimal control policies through interaction with the environment.

---

## 🧠 Solution Overview

The agent interacts with a simulated lunar environment and learns:
- How to control descent
- How to stabilize orientation
- How to minimize fuel usage
- How to land safely without crashing

Multiple RL approaches are implemented to study **scalability, stability, and performance trade-offs**.

---

## 🔁 High-Level Workflow

```

Reset Environment
↓
Observe State
↓
Select Action (ε-greedy)
↓
Simulate Physics
↓
Receive Reward & Next State
↓
Update Agent Policy
↓
Repeat for multiple episodes

```

---

## 🧩 Algorithms Implemented

| Algorithm | Description |
|---------|------------|
| Random Agent | Baseline for comparison |
| Q-Learning | Tabular value-based learning |
| SARSA | On-policy temporal difference learning |
| Approximate Q-Learning | Linear function approximation |
| Heuristic Approx Q-Learning | Feature-engineered RL |
| **Deep Q-Network (DQN)** | Neural network–based Q-learning |

The **Deep Q-Network (DQN)** is used as the final, scalable solution.

---

## 🤖 Environment Details

### Observation Space (8 values)
- Horizontal position
- Vertical position
- Horizontal velocity
- Vertical velocity
- Angle
- Angular velocity
- Left leg contact
- Right leg contact

### Action Space (Discrete – 4 actions)
- Do nothing
- Fire left engine
- Fire main engine
- Fire right engine

---

## 🎯 Reward Design

The reward function encourages:
- Moving closer to the landing pad
- Reducing velocity and angle
- Stable leg contact
- Fuel efficiency

Penalties are given for:
- Crashing
- Excessive fuel usage
- Leaving the landing zone

Reward shaping is crucial for faster and stable convergence.

---

## 🛠️ Tech Stack

- **Python**
- **OpenAI Gym**
- **Box2D** (Physics simulation)
- **PyTorch** (Deep learning)
- **NumPy**
- **Matplotlib**
- **Pygame** (Rendering)

---

## 📂 Project Structure

```

.
├── agents.py              # All RL agent implementations
├── lunar_lander.py        # Custom Lunar Lander environment
├── trainLander.py         # Training & evaluation pipeline
├── sarsa_lander.py        # SARSA implementation
├── simplegame.py          # 1D toy environment for debugging
├── plot.ipynb             # Training visualization
├── saved_results/         # Saved models & outputs
├── results/               # Reward and performance logs
└── README.md

````

---

## ▶️ How to Run

### Install Dependencies
```bash
pip install gym box2d-py torch numpy matplotlib pygame
````

### Train an Agent

```bash
python trainLander.py
```

### Run Heuristic Demo

```bash
python lunar_lander.py
```

### Evaluate a Trained Model

```bash
python trainLander.py
```

---

## 📈 Results

* Random agent fails consistently
* Tabular methods struggle due to large state space
* Approximate Q-learning improves generalization
* **Deep Q-Learning achieves stable and consistent landings**

Performance is visualized using reward and step plots.

---

## 🌍 Real-World Applications

* Autonomous spacecraft landing
* Drone and UAV control
* Robotics and motion planning
* Autonomous vehicles
* Industrial automation
* Game AI
* Reinforcement learning research & education

---

## 🚀 Key Learnings

* Reinforcement Learning fundamentals
* Value-based vs deep RL methods
* Reward shaping importance
* Experience replay for stability
* Handling continuous state spaces
* Physics-based control problems

---

## 🏁 Conclusion

This project demonstrates how **Reinforcement Learning**, especially **Deep Q-Networks**, can solve complex control problems where explicit rules or labeled data are unavailable.
It highlights scalability, adaptability, and real-world applicability of modern RL techniques.

---

## 👤 Author

Umang
AI / ML Enthusiast

---

## ⭐ Acknowledgements

* OpenAI Gym
* Box2D Physics Engine
* PyTorch Community


Just tell me 👍
```
