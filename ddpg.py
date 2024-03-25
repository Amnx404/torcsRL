import torch
import numpy as np
from torch.optim import Adam
from gym_torcs import TorcsEnv
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import argparse

def playGame(train_indicator=0, model_dir='./models'):
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    LRA = 0.0001    # Learning rate for Actor
    LRC = 0.001     # Learning rate for Critic
    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1

    action_dim = 3  # Steering/Acceleration/Brake
    state_dim = 29  # Sensors input

    np.random.seed(1337)
    torch.manual_seed(1337)

    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim, action_dim)
    actor_optimizer = Adam(actor.parameters(), lr=LRA)
    critic_optimizer = Adam(critic.parameters(), lr=LRC)
    OU_process = OU()  # Ornstein-Uhlenbeck Process
    buff = ReplayBuffer(BUFFER_SIZE)

    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    if train_indicator == 0:  # Load model weights for evaluation
        actor.load_state_dict(torch.load(f'{model_dir}/actormodel.pth'))
        critic.load_state_dict(torch.load(f'{model_dir}/criticmodel.pth'))
        actor.eval()
        critic.eval()

    print("TORCS Experiment Start.")
    for episode in range(episode_count):
        print(f"Starting Episode {episode}")
        ob = env.reset(relaunch=(episode % 3 == 0))  # Relaunch TORCS every 3 episodes
        state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        total_reward = 0.
        for step in range(max_steps):
            print("step:",step)

            #loss = 0

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = actor(state_tensor).detach().cpu().numpy()
            
            epsilon -= 1.0 / EXPLORE

            noise = np.zeros([1, action_dim])
            noise[0][0] = train_indicator * max(epsilon, 0) * OU_process.function(action[0][0],  0.0 , 0.60, 0.30)
            noise[0][1] = train_indicator * max(epsilon, 0) * OU_process.function(action[0][1],  0.5 , 1.00, 0.10)
            noise[0][2] = train_indicator * max(epsilon, 0) * OU_process.function(action[0][2], -0.1 , 1.00, 0.05)
                
            # action with noise
            action[0][0] += noise[0][0]
            action[0][1] += noise[0][1]
            action[0][2] += noise[0][2]

            next_ob, reward, done, _ = env.step(action[0])
            next_state = np.hstack((next_ob.angle, next_ob.track, next_ob.trackPos, next_ob.speedX, next_ob.speedY, next_ob.speedZ, next_ob.wheelSpinVel / 100.0, next_ob.rpm))

            buff.add(state, action[0], reward, next_state, done)

            if train_indicator and buff.count() > BATCH_SIZE:
                batch = buff.getBatch(BATCH_SIZE)
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

                # Convert to tensors
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)
                # print('states:',state)
                # print('rewards:',rewards)


                # Compute target Q values
                next_actions = actor(next_states)
                next_Q = critic(next_states, next_actions.detach())
                target_Q = rewards + GAMMA * next_Q * (1 - dones)

                # Update Critic
                current_Q = critic(states, actions)
                critic_loss = torch.mean((current_Q - target_Q.detach()) ** 2)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Update Actor
                policy_loss = -critic(states, actor(states)).mean()

                actor_optimizer.zero_grad()
                policy_loss.backward()
                actor_optimizer.step()

            total_reward += reward
            state = next_state

            if done:
                break

        if episode % 100 == 0 and train_indicator:
            torch.save(actor.state_dict(), f'{model_dir}/actormodel.pth')
            torch.save(critic.state_dict(), f'{model_dir}/`criticmodel.pth')
            print(f"Models saved at episode {episode}")
            
    env.end()
    print("Experiment Ended.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1, help='Train the model')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save models')
    args = parser.parse_args()
    playGame(args.train, args.model_dir)
    