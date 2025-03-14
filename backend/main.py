﻿from src.config import Config 
from src.utils.train_network import train_network
from src.utils.run_selfplay import run_selfplay
from src.utils.replay_buffer import ReplayBuffer
from src.utils.shared_storage import SharedStorage
import time
from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np

# MuZero training is split into two independent parts: 
# Network training and self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.

def muzero(config: Config):
    
    storage = SharedStorage()
    replay_buffer = ReplayBuffer()
    
    rewards = []
    losses = []
    moving_averages = []
    
    t = time.time()

    for i in range(config.training_episodes):
        
        # self-play
        run_selfplay(config, storage, replay_buffer)
        
        # print and plot rewards
        game = replay_buffer.last_game()
        reward_e = game.total_rewards()
        rewards.append(reward_e)
        moving_averages.append(np.mean(rewards[-20:]))
        
        # ??????
        #for _ in range(10):
            #clear_output(wait=True)
                
        print('Episode ' + str(i+1) + ' ' + 'reward: ' + str(reward_e))
        print('Moving Average (20): ' + str(np.mean(rewards[-20:])))
        print('Moving Average (100): ' + str(np.mean(rewards[-100:])))
        print('Moving Average: ' + str(np.mean(rewards)))
        print('Elapsed time: ' + str((time.time() - t) / 60) + ' minutes')       
        
        """
        plt.plot(rewards)
        plt.plot(moving_averages)
        plt.show()
        """

        # training
        loss = train_network(config, storage, replay_buffer, i).numpy()[0]
                
        # print and plot loss
        print('Loss: ' + str(loss))
        losses.append(loss)
        plt.plot(losses)
        plt.show()        
        
### Entry-point function
muzero(Config())