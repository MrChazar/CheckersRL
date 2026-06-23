from deepdraughts.env.py_env.env_utils import enable_endgame_database, get_endgame_database, set_endgame_database, \
    NO_WINNER
from .env import Game, game_status_to_str, game_is_drawn, game_is_over, game_winner
from .dqn import DQNAgent
import numpy as np
import torch
import pickle
import time
import copy
from .env import *

GAME_WIN = 1.
GAME_TIE = 0.
GAME_LOSS = -1.
PIECE_TAKEN = .01
#KING_TAKEN = 8.

class GameCollector():
    @classmethod
    def get_number_of_taken_pieces(cls, game, training_side):
        pieces = game.current_board.get_pieces()
        opponent_pieces = len([x for x in pieces if x.player != training_side])
        dqn_pieces = len(pieces) - opponent_pieces
        pieces_taken = (12 - opponent_pieces) - (12 - dqn_pieces)
        return pieces_taken

    @classmethod
    def self_play(
            cls,
            shared_net,
            epsilon,
            game_args=dict(),
            shared_database=None,
            training_side=WHITE,
            n_steps=1,
            gamma=0.99,
            device='cpu'
    ):
        """
        Play one game using DQN Agent (Self-play).

        Returns:
            transitions, winner

        Each transition:
            (
                s_board,
                s_state,
                action_id,
                n_step_reward,
                next_board,
                next_state,
                done,
                next_legal_mask
            )

        next_legal_mask:
            np.ndarray of shape [n_actions]
            True  -> legal action in next_state
            False -> illegal action in next_state
        """

        np.random.seed()

        if shared_database is not None:
            enable_endgame_database(shared_database)

        agent = DQNAgent(
            shared_net,
            epsilon=epsilon,
            device=device
        )

        game = Game(**game_args)

        # Get number of actions from network
        if hasattr(shared_net, "n_actions"):
            n_actions = shared_net.n_actions
        elif hasattr(shared_net, "module") and hasattr(shared_net.module, "n_actions"):
            n_actions = shared_net.module.n_actions
        else:
            raise ValueError("Cannot determine n_actions from shared_net.")

        def build_legal_mask(moves):
            mask = np.zeros(n_actions, dtype=np.bool_)

            for move in moves:
                move_id = move.id()

                if move_id < 0 or move_id >= n_actions:
                    raise ValueError(
                        f"move.id()={move_id} is outside valid range 0..{n_actions - 1}"
                    )

                mask[move_id] = True

            return mask

        # Temp storage
        states = []
        actions = []
        players = []
        rewards = []
        legal_masks = []

        winner = NO_WINNER

        while True:
            # Current state
            vec_board, vec_state = game.to_vector()

            available_moves = game.get_all_available_moves()
            if not available_moves:
                # This should usually be handled by game status,
                # but this prevents crashing on move.id().
                winner = -game.current_player
                break

            current_legal_mask = build_legal_mask(available_moves)

            states.append((vec_board, vec_state))
            players.append(game.current_player)
            legal_masks.append(current_legal_mask)

            # Agent selects action
            move, _ = agent.get_action(game, game.current_player)
            if move is None:
                winner = -game.current_player
                break

            actions.append(move.id())

            # Apply move
            pieces_before_move = cls.get_number_of_taken_pieces(game, training_side)

            game_status = game.do_move(move)

            pieces_after_move = cls.get_number_of_taken_pieces(game, training_side)

            # Reward / penalty for taking or losing pieces
            rewards.append((pieces_after_move - pieces_before_move) * PIECE_TAKEN)

            if game_is_over(game_status):
                winner = game_winner(game_status)

                if winner == NO_WINNER:
                    rewards[-1] += GAME_TIE
                elif winner == training_side:
                    rewards[-1] += GAME_WIN
                else:
                    rewards[-1] += GAME_LOSS

                break

        transitions = []

        total_moves = len(actions)
        if total_moves == 0:
            return transitions, winner

        dones = [False] * total_moves
        dones[-1] = True

        terminal_board = np.zeros_like(states[0][0])
        terminal_state = np.zeros_like(states[0][1])
        terminal_legal_mask = np.zeros(n_actions, dtype=np.bool_)

        for t in range(total_moves):
            R = 0.0
            done_n = False
            last_step = 0

            for step in range(n_steps):
                idx = t + step

                if idx >= total_moves:
                    break

                R += (gamma ** step) * rewards[idx]
                last_step = step

                if dones[idx]:
                    done_n = True
                    break

            next_idx = t + last_step + 1

            if next_idx < total_moves and not done_n:
                ns_board, ns_state = states[next_idx]
                ns_legal_mask = legal_masks[next_idx]
            else:
                ns_board = terminal_board
                ns_state = terminal_state
                ns_legal_mask = terminal_legal_mask

            s_board, s_state = states[t]
            action_id = actions[t]

            transitions.append(
                (
                    s_board,
                    s_state,
                    action_id,
                    R,
                    ns_board,
                    ns_state,
                    done_n,
                    ns_legal_mask
                )
            )

        return transitions, winner

    @classmethod
    def count_pieces(cls, board_tensor, side):
        # Count opponent pieces
        if side == WHITE:
            return np.sum(board_tensor[0]) + np.sum(board_tensor[1])
        else:  # BLACK
            return np.sum(board_tensor[2]) + np.sum(board_tensor[3])

    @classmethod
    def parallel_collect_selfplay(cls, n_cores, shared_model, epsilon, batch_size,
                                  game_args=dict(), filepath=None, training_side=WHITE, n_steps=1, gamma=0.99, device='cpu'):
        """
        Runs multiple self-play games in parallel to fill the buffer.
        """
        shared_database = get_endgame_database()
        try:
            from torch.multiprocessing import Pool
        except ImportError:
            from multiprocessing import Pool

        # Important: There was a mismatch between model in gpu and cpu this line helps
        #shared_model.to(device)
        shared_model.share_memory()

        with Pool(n_cores) as pool:
            game_args_list = [(shared_model, epsilon, game_args, shared_database, training_side, n_steps, gamma,
                               device)] * batch_size
            results = pool.starmap(cls.self_play, game_args_list)

            # We run n_cores * k games.
            # n_games_to_play = batch_size  # batch_size is a of num_games !
            # for _ in range(n_games_to_play):
            #     res = pool.apply_async(cls.self_play, (shared_model, epsilon, game_args, shared_database, training_side, n_steps, gamma, device) )
            #     results.append(res)
            # pool.close()
            # pool.join()

        all_transitions = []
        winners = []
        for res in results:
            transitions, winner = res
            all_transitions.extend(transitions)
            winners.append(winner)

        if filepath:
            cls.dump_data(all_transitions, filepath)

        return all_transitions, winners

    @classmethod
    def eval(cls, current_net, eval_net, i, game_args=dict(), shared_database=None, device='cpu'):
        """Evaluation: DQN vs DQN """
        np.random.seed()
        if shared_database is not None:
            enable_endgame_database(shared_database)

        agent_current = DQNAgent(current_net, epsilon=0.0, device=device)  # epsilon 0 for better eval
        agent_eval = DQNAgent(eval_net, epsilon=0.0, device=device) if eval_net else None

        game = Game(**game_args)
        white_player = agent_current if i % 2 == 0 else agent_eval
        black_player = agent_eval if i % 2 == 0 else agent_current
        WHITE = game.current_player  # Assuming starts with white

        while not game_is_over(game.query_game_status()):
            if game.current_player == WHITE:
                cur = white_player
            else:
                cur = black_player

            if cur:
                move, _ = cur.get_action(game)
            else:
                # Fallback to random if no opponent provided
                avail = game.get_all_available_moves()
                move = avail[np.random.randint(len(avail))]

            game.do_move(move)

        return game_winner(game.query_game_status())

    @classmethod
    def parallel_eval(cls, current_model, eval_model, n_cores, n_games, game_args=dict()):
        # Helper to run eval in parallel
        shared_database = get_endgame_database()
        try:
            from torch.multiprocessing import Pool
        except ImportError:
            from multiprocessing import Pool

        #current_model.cpu()
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
            # Logic needs to match env utils. Assuming result is player ID.
            # Here simplified: Just count valid completions
            wins += 1  # Placeholder

        return 0.5  # Placeholder return

    @classmethod
    def dump_data(cls, data, filepath):
        with open(filepath, "wb") as wfp:
            pickle.dump(data, wfp)