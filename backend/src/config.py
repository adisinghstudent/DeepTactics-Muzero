class Config:
    def __init__(
        self,
        action_space: int = 18, # 18 legal actions in atari
        input_planes: int = 128, # 3 rbg planes * 32 last states + 32 last actions
        height: int = 96, # Pixel height and with
        width: int = 96,
        num_input_moves: int = 32, # Number of moves that is used as input to representation model
        max_moves: float = 50_000, # Max moves before game ends
        game_name: str = "ALE/Breakout-v5",
        num_selfplay_games = 1_000_000,
        max_replay_games = 125_000, # Replay buffer size
        n_tree_searches = 50,
        epsilon: float = 0.001,
        discount: float = 0.997,
        c1: float = 1.25,
        c2: float = 19652,
        diriclet_noise = 0.25,
        dirichlet_exploaration_factor = 0.25, # Set this to 0 for deterministic prior probabilites
        batch_size = 2048,
        encode_game_state_fn = encode_state_atari,
        softmax_policy_fn = softmax_policy_atari_train,
        info_print_rate = 10,
        training_interval = 1_000,
        num_training_rolluts = 5,
        model_load_filename = "test",
        model_save_filename = "test",
    ):
        # Only to keep the type checker happy
        gym.register_envs(ale_py)

        # Environment
        self.action_space = action_space
        self.max_moves = max_moves
        self.game_name = game_name
        self.input_planes = input_planes
        self.height = height
        self.width = width
        self.num_input_moves = num_input_moves

        # Selfplay
        self.num_selfplay_games = num_selfplay_games
        self.max_replay_games = max_replay_games
        self.n_tree_searches = n_tree_searches

        self.dirichlet_noise_alpha = diriclet_noise
        self.dirichlet_exploration_factor = dirichlet_exploaration_factor # e = 0.25 as seen in Alphago Zero paper

        # Training
        self.batch_size = batch_size
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.training_interval = training_interval
        self.model_load_filepath = "models/" + game_name + "/" + model_load_filename
        self.model_save_filepath = "models/" + game_name + "/" + model_save_filename

        self.num_training_rolluts = num_training_rolluts
    
        # PUCT parameters
        self.c1 = c1
        self.c2 = c2
        self.eps = epsilon
        self.discount = discount

        # Functions that you might want to customize for your enviroment
        self.softmax_policy_fn = softmax_policy_fn
        self.encode_game_state_fn = encode_game_state_fn

        # Logging
        self.info_print_rate = info_print_rate
        
    def init_game(self) -> gym.Env:
        pass