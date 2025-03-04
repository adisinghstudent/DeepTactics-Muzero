from utils.replay_buffer import ReplayBuffer
from src.config import Config
from src.networks.network import Network
import mcts.play_game as play_game
from utils.shared_storage import SharedStorage

def self_play(config: Config, storage: SharedStorage, 
              replayBuffer: ReplayBuffer):
    nr_of_games_to_play = 1000
    for _ in range(nr_of_games_to_play):
        network = storage.latest_network()
        game = play_game(config, network)
        replayBuffer.update_buffer(game)