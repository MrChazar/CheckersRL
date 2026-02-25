from deepdraughts.env.py_env.env_utils import enable_endgame_database, get_endgame_database, set_endgame_database
from .env import Game, game_status_to_str, game_is_drawn, game_is_over, game_winner
from .ppo import PPOAgent
import numpy as np
import pickle
import copy
from .env import *

GAME_WIN = 1.
GAME_TIE = 0.
GAME_LOSS = -1.
PIECE_TAKEN = .03


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Computes Generalized Advantage Estimation from one game
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0

    # we begin from the end
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = 0.0  # after end of the game next value is 0
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]

        # delta = r + gamma * V(s') - V(s)
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]

        # A_t = delta + gamma * lambda * A_{t+1}
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    # Returns Advantages + critics predictions
    returns = advantages + values
    return advantages.tolist(), returns.tolist()


class GameCollector():
    @classmethod
    def get_number_of_taken_pieces(cls, game, training_side):
        pieces = game.current_board.get_pieces()
        opponent_pieces = len([x for x in pieces if x.player != training_side])
        ppo_pieces = len(pieces) - opponent_pieces
        pieces_taken = (12 - opponent_pieces) - (12 - ppo_pieces)
        return pieces_taken

    @classmethod
    def self_play(cls, shared_net, game_args=dict(), shared_database=None, training_side=WHITE):
        np.random.seed()
        if shared_database is not None:
            enable_endgame_database(shared_database)

        agent = PPOAgent(shared_net, device='cpu')

        game = Game(**game_args)

        states_b, states_e, actions, logprobs, values, rewards, dones, masks = [], [], [], [], [], [], [], []

        while True:
            vec_board, vec_state = game.to_vector()

            move, log_prob, value, mask = agent.get_action(game, game.current_player, deterministic=False)

            states_b.append(vec_board)
            states_e.append(vec_state)
            actions.append(move.id())
            logprobs.append(log_prob)
            values.append(value)
            masks.append(mask)

            pieces_before_move = cls.get_number_of_taken_pieces(game, training_side)
            game_status = game.do_move(move)
            pieces_after_move = cls.get_number_of_taken_pieces(game, training_side)

            step_reward = (pieces_after_move - pieces_before_move) * PIECE_TAKEN
            done = game_is_over(game_status)

            if done:
                winner = game_winner(game_status)
                if winner == 0:
                    step_reward += GAME_TIE
                elif winner == training_side:
                    step_reward += GAME_WIN
                else:
                    step_reward += GAME_LOSS

            rewards.append(step_reward)
            dones.append(done)

            if done:
                break

        # after end we compute gae
        advs, rets = compute_gae(rewards, values, dones)

        return states_b, states_e, actions, logprobs, values, rewards, dones, masks, advs, rets, winner

    @classmethod
    def count_pieces(cls, board_tensor, side):
        if side == WHITE:
            return np.sum(board_tensor[0]) + np.sum(board_tensor[1])
        else:  # BLACK
            return np.sum(board_tensor[2]) + np.sum(board_tensor[3])

    @classmethod
    def parallel_collect_selfplay(cls, n_cores, shared_model, batch_size,
                                  game_args=dict(), filepath=None, training_side=WHITE):
        shared_database = get_endgame_database()
        try:
            from torch.multiprocessing import Pool
        except ImportError:
            from multiprocessing import Pool

        shared_model.cpu()

        all_rollouts = {'b': [], 's': [], 'a': [], 'lp': [], 'v': [], 'r': [], 'd': [], 'm': [], 'adv': [], 'ret': []}

        with Pool(n_cores) as pool:
            results = []
            n_games_to_play = batch_size

            for _ in range(n_games_to_play):
                res = pool.apply_async(cls.self_play, (shared_model, game_args, shared_database, training_side))
                results.append(res)

            pool.close()
            pool.join()

            winners = []
            for res in results:
                b, s, a, lp, v, r, d, m, adv, ret, winner = res.get()
                all_rollouts['b'].extend(b)
                all_rollouts['s'].extend(s)
                all_rollouts['a'].extend(a)
                all_rollouts['lp'].extend(lp)
                all_rollouts['v'].extend(v)
                all_rollouts['r'].extend(r)
                all_rollouts['d'].extend(d)
                all_rollouts['m'].extend(m)
                all_rollouts['adv'].extend(adv)
                all_rollouts['ret'].extend(ret)
                winners.append(winner)

        if filepath:
            cls.dump_data(all_rollouts, filepath)

        return all_rollouts, winners

    @classmethod
    def eval(cls, current_net, eval_net, i, game_args=dict(), shared_database=None):
        """Evaluation: PPO vs PPO """
        np.random.seed()
        if shared_database is not None:
            enable_endgame_database(shared_database)

        agent_current = PPOAgent(current_net, device='cpu')
        agent_eval = PPOAgent(eval_net, device='cpu') if eval_net else None

        game = Game(**game_args)
        white_player = agent_current if i % 2 == 0 else agent_eval
        black_player = agent_eval if i % 2 == 0 else agent_current
        WHITE = game.current_player

        while not game_is_over(game.query_game_status()):
            if game.current_player == WHITE:
                cur = white_player
            else:
                cur = black_player

            if cur:
                # we are interested only in move
                move, _, _, _ = cur.get_action(game, deterministic=True)
            else:
                avail = game.get_all_available_moves()
                move = avail[np.random.randint(len(avail))]

            game.do_move(move)

        return game_winner(game.query_game_status())

    @classmethod
    def parallel_eval(cls, current_model, eval_model, n_cores, n_games, game_args=dict()):
        shared_database = get_endgame_database()
        try:
            from torch.multiprocessing import Pool
        except ImportError:
            from multiprocessing import Pool

        current_model.cpu()
        if eval_model: eval_model.cpu()

        with Pool(n_cores) as pool:
            results = []
            for i in range(n_games):
                res = pool.apply_async(cls.eval, (current_model.policy_net,
                                                  eval_model.policy_net if eval_model else None,
                                                  i, game_args, shared_database))
                results.append(res)
            pool.close()
            pool.join()

            raw_results = [r.get() for r in results]

        wins = 0
        for i, res in enumerate(raw_results):
            wins += 1

        return 0.5

    @classmethod
    def dump_data(cls, data, filepath):
        with open(filepath, "wb") as wfp:
            pickle.dump(data, wfp)