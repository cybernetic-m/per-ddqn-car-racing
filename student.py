import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import namedtuple, deque
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

def plot_training_metrics(training_reward, mean_training_reward):
    plt.figure(figsize=(8, 6))
    
    # Plot Training and Mean Rewards
    plt.plot(training_reward, label='Training Rewards', color='blue', alpha=0.4)
    plt.plot(mean_training_reward, label='Mean Rewards', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Training and Mean Rewards')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

##---- Utils Functions and Classes ----##
def transform(img):
    img = img[0:84, 6:90] # Crop the image in order to eliminate useless details
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0 # Transform the image in grayscale and normalize pixels
    return img
           
##---- DQN MODELS ----##
class DQN(nn.Module):
    def __init__(self, in_channels = 4, out_dim = 5):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4)  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)  
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, out_dim)

    def forward(self, x):
        # Handle single sample or batch (populate of the buffer)
        if x.dim() == 3:  # Single sample: [4, 84, 84]
            x = x.unsqueeze(0)  # Add batch dimension -> [1, 4, 84, 84]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flattening
        x = x.view(x.size(0), -1)  # Flatten from [batch_size, channels, height, width]
        # Fully connected layers
        x = self.fc1(x)
        out = self.fc2(x)
        return out 

class Q_Net(nn.Module):
    def __init__(self, lr):
        super(Q_Net, self).__init__()
        
        self.q_net = DQN()
    
        print("Q_network: ", self.q_net)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def get_qvals(self, state):
        q_vals = self.q_net(state)
        return q_vals

    def greedy_action(self, state): 
        
        q_vals = self.get_qvals(state)
        greedy_a = torch.max(q_vals, dim=-1)[1].item()
        
        return greedy_a

##---- Prioritized Experience Replay Buffer ----##
class PER_Buffer():
    def __init__(self, memory_size = 50000, burn_in = 10000, alpha = 0.1, eps = 1e-2, beta = 0.1):
        self.memory_size = memory_size # Limit size of the replay buffer
        self.burn_in = burn_in # Initial size population in the replay buffer
        self.alpha = alpha # Exponent to compute priorities
        self.eps = eps # Constant to add to TD error
        self.beta = beta # Exponent to compute weights
        self.Transitions = namedtuple('Transition', 
                                 field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.transitions = deque(maxlen=memory_size) # buffer of transitions
        self.priorities = deque(maxlen=memory_size) # buffer of priorities
        self.max_priority = eps # Initialize max priority to epsilon (max_priority is the highest value of priorities that will be assigned to new added sample in a first time)

    def burn_in_capacity(self):
        return (len(self.transitions) / self.burn_in)*100 # return the percentage of capacity of the burn_in part of the replay buffer
    def capacity(self):
        return (len(self.transitions) / self.memory_size)*100 # return the percentage of capacity of the entire replay buffer

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if isinstance(priority, np.ndarray):
                priority = priority.item()  # Convert NumPy array to scalar
            self.priorities[idx] = (priority+self.eps) ** self.alpha 
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def add(self, state, action, reward, done, next_state):
        # Add Transition
        self.transitions.append(self.Transitions(state, action, reward, done, next_state)) 
        # Add Priority 
        self.priorities.append(float(self.max_priority))
    
    def sample(self, batch_size):
        # Compute the probabilities of sampling
        priorities = np.array(self.priorities)
        probs = priorities/priorities.sum()

        # Generate indices (batch_size number of indices)  based on probabilities of each transition
        indices = np.random.choice(len(self.priorities), size=batch_size, p=probs) 
        
        
        batch = [self.transitions[i] for i in indices]

        # Compute of weights
        weights = (probs[indices] * len(self.transitions)) ** (-self.beta)
        weights /= weights.max()  # Normalize weights 

        return batch, indices, weights

##---- Policy Class ----##

class Policy(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()

        # Hyperparam
        self.continuous = False
        self.batch_size = 64
        self.epsilon = 0.8 # Epsilon for the greedy action
        self.max_ep = 2000 # Set max_episodes, the training is stopped after max_ep
        self.memory_size = 100000 # Size of the replay buffer
        self.burn_in = 20000 # Initial population of the replay buffer
        self.eps = 1e-2
        self.alpha = 0.1
        self.beta = 0.1
        self.gamma = 0.99 # Discount factor
        self.lr = 1e-4 # Learning rate
        self.epsilon_decay = 0.99
        self.network_update_frequency = 10 # Number of episodes to update the Q_network
        self.network_sync_frequency = 50 # Number of episodes to copy the Q network states into the target network
        self.render = False # change if you want to render

        self.device = device
        self.skip_frames = 50 # Number of frames to skip (initial zoom of the racing game)
        self.stack_dim = 4 # Number of frames to stack in batch
        self.last_frames =np.zeros((self.stack_dim, *(84,84)), dtype=np.float32)
        self.count_frames = 0 # Counter of frames to check if we have exceed the first 50 frames to skip
        self.stack = np.zeros((self.stack_dim, 84, 84)) # Initialization of stack of images [4, 84, 84]
        self.rewards = 0
        
        if self.render:
            self.env = gym.make('CarRacing-v2', continuous=False, render_mode='human')
        else:
            self.env = gym.make('CarRacing-v2', continuous=False)
    
        self.done = False
        self.max_patience = 100
        self.actual_patience = 0
        
        # Network and Target Network
        self.network = Q_Net(lr = self.lr).to(self.device)
        self.target_network = Q_Net(lr = self.lr).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
    
        # Buffer
        self.buffer = PER_Buffer(memory_size=self.memory_size, burn_in=self.burn_in, eps=self.eps, alpha=self.alpha, beta=self.beta)

        # Set the devide mode on GPU (if available CUDA for Nvidia and  MPS for Apple Silicon) or CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Function that return the best action 
    def act(self, state):
        # Return "do nothing" for the first 50 - 4 = 46 frames (because of the zoom)
        if self.count_frames < self.skip_frames - self.stack_dim:
            self.count_frames += 1 # Increment the counter
            return 0 

        # Return "do nothing" frames between 46,50 but saving the 4 frames in between
        if self.count_frames < self.skip_frames:
            s=transform(state)
            j=self.count_frames%(self.skip_frames - self.stack_dim)
            self.last_frames[j]=s
            self.count_frames+=1
            return 0

        s=transform(state)
        self.last_frames[:-1] = self.last_frames[1:] # [a,b,c,d] => [b,c,d, *]
        self.last_frames[-1] = s  # Insert the new image at the end of the list [b,c,d, *] => [b,c,d,e]
        action = self.network.greedy_action(torch.FloatTensor(self.last_frames).to(self.device)) # Return the best action
        return action


    def take_step(self, explore):
        self.step_count += 1 # Increment the counter because of a new step is done

        # epsilon greedy policy used in training 
        if explore:
            action = self.env.action_space.sample() # Used to populate the replay buffer
        else:
            p = np.random.random() # Generate a random number between [0,1]
            if p < self.epsilon:
                action = self.env.action_space.sample() # Explore
            else: 
                action = self.network.greedy_action(torch.FloatTensor(self.s_0).to(self.device)) # Exploit
        
        truncation=False

        # Simulate the action
        s_1, reward, terminated, truncated, info = self.env.step(action)
        s_1 = transform(s_1)
        self.last_frames[:-1] = self.last_frames[1:]   # [a,b,c,d] => [b,c,d, *]
        self.last_frames[-1] = copy.deepcopy(s_1)   # Insert the new image at the end of the list [b,c,d, *] => [b,c,d,e]

        if reward >=0:
            self.actual_patience = 0
        else:
            self.actual_patience+= 1
            if self.actual_patience == self.max_patience:
                truncation = True
                self.actual_patience = 0
                reward=-100

        # Append experience in the buffer
        self.buffer.add(copy.deepcopy(self.s_0), action, reward, int(terminated), copy.deepcopy(self.last_frames))

        self.rewards += reward

        self.s_0 = copy.deepcopy(self.last_frames)
        
        if terminated or truncation:
            done = True # Done is true if terminated or truncated are true
            self.actual_patience = 0
            self.env = self.my_reset(self.env)
        else:
            done = False 
            
        return done
    
    def my_reset(self, env):
        self.count_frames=0
        _,_=env.reset()
        for i in range(self.skip_frames):
            if self.count_frames < self.skip_frames - self.stack_dim:
                s, r, terminated, truncated, info = env.step(0)
                self.count_frames+=1
                continue
        
            if self.count_frames < self.skip_frames:
                s, r, terminated, truncated, info = env.step(0)
                s=transform(s)
                j=self.count_frames%(self.skip_frames - self.stack_dim)
                self.last_frames[j]=copy.deepcopy(s)
                self.count_frames+=1
                continue
        
        self.s_0=copy.deepcopy(self.last_frames)
        return env           


    # Training Function
    def train(self):
        # Initialization of all the list needed to track the train
        self.training_reward = []
        self.training_loss = []
        self.mean_training_loss = []
        self.mean_training_reward = []
        self.step_count = 0
        best_reward = -1000000
        
       
        self.env = self.my_reset(self.env)

        # Populate buffer util the burn-in capacity is reached
        print ("Populating the buffer...")
        with tqdm(total=100, desc="Buffer Burn-In", unit="%", leave=True) as pbar:
            prev_progress = 0
            while self.buffer.burn_in_capacity() < 100:
                self.take_step(explore=True)
                current_progress = int(self.buffer.burn_in_capacity())
                pbar.update(current_progress - prev_progress)
                prev_progress = current_progress
        print("Burn in capacity full: ", self.buffer.burn_in_capacity(), "%")
            
        ep = 1 # Initialization of episodes (counter) 
        training = True # Set training True, until max_episodes is reached
        
        print("Training...")
        while training:
        
            self.rewards = 0 # Initialize the reward to zero (to accumulate each)
            done = False # Set to false (while done = False the agent take steps)
            
            # Start the loop until the agent reach a terminated or truncated state
            while not done:
                
                done = self.take_step(explore=False) # Take an action following the epsilon greedy policy
                
                # Each 10 episodes update the Q_net weights
                # Each 50 episodes copy Q_net weights into the Target Network
                # Update Network
                if self.step_count % self.network_update_frequency == 0:
                    self.network.optimizer.zero_grad() # Zeros the gradient accumulated before
                    batch, indices, weights = self.buffer.sample(self.batch_size)

                    # Convert states, actions ... into torch tensors for each sample in the batch_size
                    states = torch.stack([torch.from_numpy(batch[trans][0]).float() for trans in range(self.batch_size)], dim=0)
                    actions = torch.tensor([batch[trans][1] for trans in range(self.batch_size)], dtype=torch.int64).unsqueeze(1)
                    rewards = torch.tensor([batch[trans][2] for trans in range(self.batch_size)], dtype=torch.float32).unsqueeze(1)
                    dones = torch.tensor([batch[trans][3] for trans in range(self.batch_size)], dtype=torch.int8).unsqueeze(1)
                    next_states = torch.stack([torch.from_numpy(batch[trans][4]).float() for trans in range(self.batch_size)], dim=0)
                    
                    # Compute q_vals
                    q_vals = self.network.get_qvals(states) # shape [64, 5]
                    # Take the q-values corresponding to the actions
                    # q_vals = [[1,2,3,4,5], [9, 8, 7, 6, 5], ...], actions = [[1], [0], ..] -> after gather [[2], [9], ...] 
                    # actions shape [64] -> after unsqueeze [64,1] same shape of q_vals needed for torch gather
                    q_vals = torch.gather(q_vals, 1, actions) 

                    # Compute target q_vals of next state
                    with torch.no_grad():
                        next_action = torch.argmax(self.network.get_qvals(next_states), dim=1).unsqueeze(1)
                        next_q_vals = self.target_network.get_qvals(next_states).gather(1,next_action) # shape [64, 5]
                        # Take the maximum q_value along dim 1 (i.e. the 5 values of 64 batches)
                        # target_q_vals = [[1,2,3,4,5], [9, 8, 7, 6, 5], ...] -> [[5], [9], ...]
                        # Take the [0] element because torch.max return both elements and indices of the max elements, 
                        # I take only the tensor of values!
                        # I reshape with view to obtain [64,1] shape!
                        #target_q_vals = torch.max(target_q_vals, dim=1)[0].view(-1,1)
                        target_q_vals = rewards + (1-dones)*self.gamma*next_q_vals
                    
                    # Compute td error
                    td_errors = torch.abs(target_q_vals - q_vals) # Absolute value as in the algorithm

                    # Update priorities in the buffer
                    #for i in range(self.batch_size):
                    self.buffer.update_priorities(indices, td_errors.detach().numpy())
                    
                    # Weights in tensor [64,1]
                    weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

                    # Compute the Mean Square Error Loss (weighted because of PER)
                    loss = (weights * td_errors.pow(2)).mean()
                    # Compute gradient
                    loss.backward()

                    # Update the weights of Q Net
                    self.network.optimizer.step()

                
                if self.step_count % self.network_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

                # If the agent is in a termination or truncation state
                if done:
                    if self.epsilon >= 0.05:
                        self.epsilon = self.epsilon * self.epsilon_decay
                    ep += 1 # Count one episode
                
                    self.training_reward.append(self.rewards)
                    self.training_loss.append(loss.item())
                    mean_rewards = np.mean(self.training_reward[-50:])
                    mean_loss = np.mean(self.training_loss[-50:])
                    self.mean_training_reward.append(mean_rewards)
                    self.mean_training_loss.append(mean_loss)

                    print(
                        "\rEpisode {:d} Episode Reward = {:.2f} Episode Loss = {:.2f}  Mean Rewards {:.2f} Mean Loss = {:.2f}\t\t".format(
                            ep, self.rewards, loss, mean_rewards, mean_loss), end="")
                    print()
                    if self.rewards > best_reward:
                        best_reward = self.rewards
                        self.save_best()
                        print("Saved model at ep:", ep, " with reward:", best_reward)

                    # Check if the number of episodes is the maximum, in this case stop the training
                    if ep >= self.max_ep:
                        training = False
                        self.save()
                        print("Limit of episodes reached: ", self.max_ep)
                        print("Saved the last model as 'model_last.pt': ")
                        plot_training_metrics(self.training_reward, self.mean_training_reward)
                        break

                    
    # Function to save the last model
    def save(self):
        torch.save(self.network.q_net.state_dict(), 'model.pt')

    # Function to save the best model
    def save_best(self):
        torch.save(self.network.q_net.state_dict(), 'model_best.pt')
 
    def load(self):
        self.network.q_net.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret





