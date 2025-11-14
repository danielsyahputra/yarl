import numpy as np
import matplotlib.pyplot as plt

class MAB:
    def __init__(self, n_machines: int = 10) -> None:
        self.true_rewards = np.random.normal(0, 1, n_machines)
        self.n_machines = n_machines 
        print(self.true_rewards)
        print(f"Best machines: #{np.argmax(self.true_rewards)} with reward {np.max(self.true_rewards)}")

    def pull(self, machine):
        return np.random.normal(self.true_rewards[machine], 1)

class EpsilonGreedyA:
    def __init__(self, n_machines: int = 10, epsilon :float = 0.1, learning_rate: float = 0.1) -> None:
        self.n_machines = n_machines
        self.epsilon = epsilon
        self.learning_rate = learning_rate 

        self.Q_s = np.zeros(self.n_machines)
        self.machine_counts = np.zeros(self.n_machines)

    def select_action(self):
        # explore path
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_machines)
        # exploitation path
        return np.argmax(self.Q_s)
    
    def update(self, machine, reward):
        self.machine_counts[machine] += 1
        self.Q_s[machine] += self.learning_rate * (reward - self.Q_s[machine])

def run_mab(n_machines: int = 10, n_steps : int = 1000, epsilon : float = 0.1, learning_rate: float = 0.1, random_seed : int = 42):
    np.random.seed(random_seed)
    mab = MAB(n_machines)
    agent = EpsilonGreedyA(n_machines, epsilon=epsilon, learning_rate=learning_rate)
    rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps)

    optimal_machine = np.argmax(mab.true_rewards)

    for step in range(n_steps):
        machine = agent.select_action()
        
        optimal_actions[step] = 1 if machine == optimal_machine else 0
        
        reward = mab.pull(machine)
        rewards[step] = reward
        
        agent.update(machine, reward)

    cumulative_average_reward = np.cumsum(rewards) / (np.arange(n_steps) + 1)
    
    optimal_action_percentage = np.cumsum(optimal_actions) / (np.arange(n_steps) + 1)
    
    return {
        'rewards': rewards,
        'cumulative_average_reward': cumulative_average_reward,
        'optimal_action_percentage': optimal_action_percentage,
        'agent': agent,
        'bandit': mab,
        'optimal_machine': optimal_machine 
    }

def visualize_results(results):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(results['cumulative_average_reward'])
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(results['optimal_action_percentage'])
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action %')
    plt.title('Frequency of Optimal Action Selection')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nFinal Results:")
    print(f"Agent's final machine value estimates: {results['agent'].Q_s.round(3)}")
    print(f"True reward values: {results['bandit'].true_rewards.round(3)}")
    print(f"Optimal machine: {results['optimal_machine']}")
    print(f"Times each machine was pulled: {results['agent'].machine_counts.astype(int)}")
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(results['agent'].machine_counts)), results['agent'].machine_counts)
    plt.xlabel('Machine')
    plt.ylabel('Number of Pulls')
    plt.title('Distribution of Machine Selections')
    plt.show()

results = run_mab(n_machines=10, n_steps=1000, epsilon=0.1)
visualize_results(results)

def compare_epsilons():
    epsilons = [0.01, 0.1, 0.5]
    plt.figure(figsize=(15, 6))
    
    for i, epsilon in enumerate(epsilons):
        results = run_mab(epsilon=epsilon, random_seed=42)
        plt.subplot(1, 2, 1)
        plt.plot(results['cumulative_average_reward'], label=f'ε={epsilon}')
        plt.subplot(1, 2, 2)
        plt.plot(results['optimal_action_percentage'], label=f'ε={epsilon}')
    
    plt.subplot(1, 2, 1)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action %')
    plt.title('Frequency of Optimal Action')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

compare_epsilons()
