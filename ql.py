import numpy as np
import random


class Node:
    def __init__(self, id):
        self.connected_nodes = []  
        self.q_values = []
        self.travel_times = []

        self.id  = id

class Graph:
    def __init__(self, nodes) -> None:
        self.nof_nodes = len(nodes)
        self.nodes = nodes

        # create the circular graph
        l1 = self.nodes
        l2 = [self.nodes[-1]] + self.nodes[:-1]
        for n1, n2 in zip(l1,l2):
            n1.connected_nodes.append(n2)

        l1 = self.nodes
        l2 = self.nodes[1:] + [self.nodes[0]]
        for n1, n2 in zip(l1,l2):
            n1.connected_nodes.append(n2)

    def add_connections(self):
        temp = random.choice(self.nodes)
        # buat random shortcut
        for _ in range(self.nof_nodes):
            while len(temp.connected_nodes) < 3:
                # Select from available nodes
                node_options = [node for node in self.nodes if len(node.connected_nodes) < 3 and node != temp and node not in temp.connected_nodes]

                if not node_options:
                    break

                node_selection = random.choice(node_options)
                temp.connected_nodes.append(node_selection)
                node_selection.connected_nodes.append(temp)

            # Move to the next node in the chain
            temp = temp.connected_nodes[1]

    def populate_travel_times_and_q_values(self):

        temp = random.choice(self.nodes)
        for _ in range(self.nof_nodes):
            if(len(temp.connected_nodes)==2):
                travel_times_p1 = [random.randint(1, 3) for _ in range(2)]
                travel_times_p2 = [item%3+1 for item in travel_times_p1]
                travel_times_p3 = [item%3+1 for item in travel_times_p2]

                temp.travel_times = [travel_times_p1, travel_times_p2, travel_times_p3]
                temp.q_values = [[0]*2]*3

            else:
                travel_times_p1 = [random.randint(1, 3) for _ in range(3)]
                travel_times_p2 = [item%3+1 for item in travel_times_p1]
                travel_times_p3 = [item%3+1 for item in travel_times_p2]

                temp.travel_times = [travel_times_p1, travel_times_p2, travel_times_p3]
                temp.q_values = [[0]*3]*3

            # Move to the next node in the chain
            temp = temp.connected_nodes[1]

class Environment:
    def __init__(self, start, goal):
        self.time_of_day = 0
        self.start = start
        self.state = start
        self.goal = goal

    def reset(self):
        self.time_of_day = 0
        self.state = self.start
        return self.state

    def step(self, action):

        if(self.time_of_day<8):
             reward = -self.state.travel_times[0][action]
             self.time_of_day = (self.time_of_day + self.state.travel_times[0][action])%24
             part_of_day =0
        elif(self.time_of_day<16):
             reward = -self.state.travel_times[1][action]
             self.time_of_day = (self.time_of_day + self.state.travel_times[1][action])%24
             part_of_day =1
        else:
             reward = -self.state.travel_times[2][action]
             self.time_of_day = (self.time_of_day + self.state.travel_times[2][action])%24
             part_of_day =2

        if(self.time_of_day<8):
            next_part_of_day = 0
        elif(self.time_of_day<16):
            next_part_of_day = 1
        else:
            next_part_of_day = 2


        self.state = self.state.connected_nodes[action]
        done = False

        if self.state == self.goal:
            reward += 100
            done = True

        return self.state, reward,part_of_day,next_part_of_day, done

class Agent:
    def __init__(self, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99965, min_epsilon=0.001):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon


    def choose_action(self, state, part_of_day):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(state.q_values[part_of_day]))
        else:
            max_value = max(state.q_values[part_of_day])
            max_indices = [i for i, value in enumerate(state.q_values[part_of_day]) if value == max_value]
            return random.choice(max_indices)


    def learn_step(self, state, action, reward, next_state, part_of_day, next_part_of_day, done):
        # change the q_value of the action that is planned to happen...
        # by using the immediate reward of the action, the best q_value of the next state...
        # and the negation of the q_value of the action that is planned to happen
        # For details, see the Learning Step illustration in the process section and end of the "My application of Q-Learning" section.
        
        max_value = max(next_state.q_values[next_part_of_day])
        max_indices = [i for i, value in enumerate(next_state.q_values[next_part_of_day]) if value == max_value]
        best_next_action = random.choice(max_indices)

        td_target = reward + self.discount_factor * next_state.q_values[next_part_of_day][best_next_action] * (not done)

        td_error = td_target - state.q_values[part_of_day][action]

        state.q_values[part_of_day][action] += self.learning_rate * td_error

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

random.seed(5)
np.random.seed(5)

nof_nodes = 100
nodes = [Node(i) for i in range(nof_nodes)]
g = Graph(nodes)

g.add_connections()
g.populate_travel_times_and_q_values()

random.seed(10)
np.random.seed(10)


start = nodes[0]
goal = nodes[int(nof_nodes/2)]
env = Environment(start,goal)
agent = Agent()


nof_episodes = 20000
path_nodes = []
time_to_traverse = []

for episode in range(nof_episodes):
    done = False
    state = env.reset()
    total_reward = 0
    part_of_day = 0

    while not done:
    
        if episode+1 == nof_episodes:
            path_nodes.append("node "+str(state.id))

        action = agent.choose_action(state, part_of_day)

        if episode+1 == nof_episodes:
            time_to_traverse.append(state.travel_times[part_of_day][action])

        next_state, reward, part_of_day, next_part_of_day, done = env.step(action)

        agent.learn_step(state, action, reward, next_state, part_of_day, next_part_of_day, done)

        state = next_state

        if(done and episode+1 == nof_episodes):
            path_nodes.append("node "+str(state.id))

        part_of_day = next_part_of_day
        total_reward += reward
        
    # For certain episodes, print total_reward of the last path taken and epsilon.
    if (episode + 1) % 1000 == 0:
        print("Episode:",episode + 1,"Total Reward:",total_reward,"Epsilon:",agent.epsilon)


for i in range(len(path_nodes)-1):
    print(path_nodes[i]+" to "+path_nodes[i+1]+" (time to traverse:"+str(time_to_traverse[i])+")")

