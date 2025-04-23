import numpy as np
from typing import Dict, Optional, Tuple, NamedTuple
from board import Board
from MoveSorter import MoveSorter
import time
import math

TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

# Sử dụng NamedTuple để làm rõ cấu trúc dữ liệu trong TT
class TTEntry(NamedTuple):
    score: int
    depth: int
    flag: int
    best_move_mask: Optional[int]

class TimeLimitExceededError(Exception):
    """Custom exception for search timeout."""
    pass

class Solver:

    def __init__(self, board_class: type[Board], max_cache_size=1_000_000):
        self.Board = board_class
        self.W = self.Board.WIDTH
        self.H = self.Board.HEIGHT

        # Counters and Timing
        self.node_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._start_time = 0.0
        self._time_limit = None
        self._time_limit_reached = False

        # Move Ordering Scores
        self.SCORE_PV_MOVE = 30000
        self.SCORE_TT_MOVE = 25000
        self.SCORE_WINNING_MOVE = 20000000
        self.SCORE_BLOCKING_MOVE = 15000000

        self.column_order = [self.W // 2 + (1 - 2 * (i % 2)) * (i + 1) // 2 for i in range(self.W)]

        # Transposition Table
        self.trans_table: Dict[int, TTEntry] = {}
        self.max_cache_size = max_cache_size

    def reset_counters(self):
        self.node_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self):
        self.trans_table.clear()
        print("Transposition Table cleared.")
    
    def _check_time_limit(self):
        # Chỉ kiểm tra và raise MỘT LẦN để tránh spam exception
        if not self._time_limit_reached and self._time_limit is not None and (time.time() - self._start_time) >= self._time_limit:
            self._time_limit_reached = True 
            raise TimeLimitExceededError()

    def solve(self, board: Board, target_depth: int, time_limit: Optional[float] = None) -> Tuple[int, Optional[int]]:
        self.reset_counters()
        self._start_time = time.time()
        self._time_limit = time_limit

        root_score, root_move = self._check_root_immediate_terminal(board)
        if root_move is not None or root_score is not None: # Nếu có kết quả ngay
             print(f"Immediate result: Score={root_score}, Move Col={self.get_col_from_move(root_move)}")
             return root_score if root_score is not None else 0, root_move

        best_score_overall = -math.inf
        best_move_overall = 0
        pv_move_from_last_iter = 0
        possible_root_moves = board.possible()

        "Iterative Deepening"
        for current_depth in range(1, target_depth + 1):
            search_start_time = time.time()
            print(f" Depth {current_depth}:", end="", flush=True)

            try:
                current_score, current_best_move = self._search_at_depth(
                    board, current_depth, possible_root_moves, pv_move_from_last_iter
                )

                if current_best_move != 0:
                    best_score_overall = current_score
                    best_move_overall = current_best_move
                    pv_move_from_last_iter = best_move_overall
                elif best_move_overall == 0 and possible_root_moves != 0:
                    best_move_overall = possible_root_moves & -possible_root_moves
                    pv_move_from_last_iter = best_move_overall

                if self._report_progress_and_check_stop(current_depth, best_score_overall, best_move_overall, search_start_time):
                    break

                if self._time_limit_reached:
                    print(f" Timeout detected after depth {current_depth} search finished.")
                    break
            except TimeLimitExceededError:
                print(f" Timeout exception reached solve loop at depth {current_depth}.")
                break 

        if best_move_overall == 0 and possible_root_moves != 0:
            best_move_overall = possible_root_moves & -possible_root_moves

        return best_score_overall, best_move_overall

    def _check_root_immediate_terminal(self, board: Board) -> Tuple[Optional[int], Optional[int]]:
        "Check if current node is a terminal"
        possible = board.possible()

        if board.nb_moves() >= self.W * self.H: return 0, None
        if possible == 0: return 0, None

        winning_moves = board.winning_position() & possible
        if winning_moves:
            win_move = winning_moves & -winning_moves
            return self.SCORE_WINNING_MOVE, win_move # TRUE
        
        return None, None # FALSE

    def _search_at_depth(self, board: Board, depth: int, possible_moves: int, pv_move: int) -> Tuple[int, int]:
        "Search at a specific depth"
        alpha = -math.inf
        beta = math.inf
        best_score = -math.inf
        best_move = 0
        move_count_root = 0

        moves = self._generate_and_sort_moves(board, possible_moves, pv_move=pv_move)
        next_move = moves.getNext()

        "Ensure that at least one move is tried"
        if not next_move and pv_move and (pv_move & possible_moves):
             next_move = pv_move

        while next_move:
            self._check_time_limit() 

            move_count_root += 1
            board_copy = board.copy()
            board_copy.play(next_move)
            score = 0

            # PVS logic
            if move_count_root == 1:
                score = -self._negamax(board_copy, -beta, -alpha, depth - 1)
            else:
                score = -self._negamax(board_copy, -alpha - 1, -alpha, depth - 1)
                if score > alpha and score < beta:
                     self._check_time_limit()
                     score = -self._negamax(board_copy, -beta, -alpha, depth - 1)

       
            if score > best_score:
                best_score = score
                best_move = next_move 
                if best_score > alpha:
                    alpha = best_score

            if alpha >= beta:
                break 
            next_move = moves.getNext()

        return best_score, best_move

    def _report_progress_and_check_stop(self, depth: int, score: int, move: int, depth_start_time: float) -> bool:
        "Logging function"
        depth_time = time.time() - depth_start_time
        total_time = time.time() - self._start_time
        move_col = self.get_col_from_move(move)
        print(f" done. Best Move: Col {move_col + 1} Score: {score}. Time: {depth_time:.2f}s (Total: {total_time:.2f}s)")

        if self._time_limit is not None and total_time >= self._time_limit:
            print(f"  Time limit ({self._time_limit}s) reached. Using results from depth {depth}.")
            return True

        max_possible_score = self.SCORE_WINNING_MOVE
        min_possible_score = -max_possible_score

        is_certain_terminal_score = False
        if isinstance(score, (int, float)) and score != -math.inf:
            if score == max_possible_score or score == min_possible_score:
            # if score == max_possible_score:
                 is_certain_terminal_score = True

        if is_certain_terminal_score:
            print(f"  Found exact terminal score ({score}) at depth {depth}. Stopping early.")
            return True

        return False 

    def _prune_cache(self):
        "refresh cache"
        if len(self.trans_table) > self.max_cache_size * 1.2:
             keys_to_remove = list(self.trans_table.keys())[:int(self.max_cache_size * 0.2)]
             for key in keys_to_remove:
                 del self.trans_table[key]
    
    def heuristic(self, board: Board) -> int:
        P = board.current_position # current player
        O = P ^ board.mask  # opponent
        M = board.mask
        E = ~M & self.Board.board_mask

        player_winning_cells = self.Board.compute_winning_position(P, M)
        opponent_winning_cells = self.Board.compute_winning_position(O, M)

        score = 0

        # mid col
        center_col_index = self.W // 2
        center_col_mask = self.Board.column_mask(center_col_index)
        center_count = self.Board.pop_count(P & center_col_mask)
        score += center_count * 3

        # 3 player - 1 empty
        player_threats_3p1e = self.Board.pop_count(player_winning_cells)
        score += player_threats_3p1e * 5

        # 3 opponent - 1 empty
        opponent_threats_3o1e = self.Board.pop_count(opponent_winning_cells)
        score -= opponent_threats_3o1e * 6

        # 2 player, 2 empty (Potential)
        patterns22_mask = np.int64(0)
        patterns22_mask_opponent = np.int64(0)
        shifts = [1, self.H, self.H + 1, self.H + 2] # dọc, chéo /, ngang, chéo \
        P_masked = P & self.Board.board_mask
        O_masked = O & self.Board.board_mask

        for s in shifts:
            P_s = (P_masked >> s) & self.Board.board_mask
            P_2s = (P_masked >> (2*s)) & self.Board.board_mask
            P_3s = (P_masked >> (3*s)) & self.Board.board_mask

            O_s = (O_masked >> s) & self.Board.board_mask
            O_2s = (O_masked >> (2*s)) & self.Board.board_mask
            O_3s = (O_masked >> (3*s)) & self.Board.board_mask

            E_s = (E >> s)
            E_2s = (E >> (2*s))
            E_3s = (E >> (3*s))

            patterns22_mask |= P_masked & P_s & E_2s & E_3s    # XX..
            patterns22_mask |= E & E_s & P_2s & P_3s           # ..XX
            patterns22_mask |= P_masked & E_s & P_2s & E_3s    # X.X.
            patterns22_mask |= E & P_s & E_2s & P_3s           # .X.X

            # Opponent patterns
            patterns22_mask_opponent |= O_masked & O_s & E_2s & E_3s
            patterns22_mask_opponent |= E & E_s & O_2s & O_3s
            patterns22_mask_opponent |= O_masked & E_s & O_2s & E_3s
            patterns22_mask_opponent |= E & O_s & E_2s & O_3s

        score_22 = self.Board.pop_count(patterns22_mask)
        score += score_22 * 2

        score_22_opponent = self.Board.pop_count(patterns22_mask_opponent)
        score -= score_22_opponent * 3

        return score

    def _generate_and_sort_moves(self, board: Board, possible_moves: int,
                                 tt_best_move: int = 0, pv_move: int = 0) -> MoveSorter:
        moves = MoveSorter(self.Board.WIDTH)

        winning_moves_now = board.winning_position() & possible_moves
        opponent_pos = board.current_position ^ board.mask
        opponent_winning_cells = self.Board.compute_winning_position(opponent_pos, board.mask)
        playable_opponent_wins = opponent_winning_cells & possible_moves
        is_must_block = (playable_opponent_wins != 0)
        move_added_flags = np.int64(0)

        "from PVS"
        if pv_move and (pv_move & possible_moves):
            score = self.SCORE_PV_MOVE
            if pv_move & winning_moves_now: score = self.SCORE_WINNING_MOVE
            elif is_must_block and (pv_move & playable_opponent_wins): score = self.SCORE_BLOCKING_MOVE
            moves.add(pv_move, score)
            move_added_flags |= pv_move

        "TT"
        if tt_best_move and (tt_best_move & possible_moves) and not (tt_best_move & move_added_flags):
            score = self.SCORE_TT_MOVE
            if tt_best_move & winning_moves_now: score = self.SCORE_WINNING_MOVE
            elif is_must_block and (tt_best_move & playable_opponent_wins): score = self.SCORE_BLOCKING_MOVE
            moves.add(tt_best_move, score)
            move_added_flags |= tt_best_move

        "win > block opp > ..."
        for i in range(self.Board.WIDTH):
            col = self.column_order[i]
            move = possible_moves & self.Board.column_mask(col)
            if move and not (move & move_added_flags):
                score = 0 # Điểm mặc định cho nước đi thường
                if move & winning_moves_now:
                    score = self.SCORE_WINNING_MOVE
                elif is_must_block and (move & playable_opponent_wins):
                    score = self.SCORE_BLOCKING_MOVE
                moves.add(move, score)

        return moves


    def _handle_tt_lookup(self, board_key: int, depth: int, alpha: int, beta: int) -> Tuple[bool, int, int, Optional[int], Optional[TTEntry]]:
        cached_entry = self.trans_table.get(board_key)
        cached_best_move = 0

        if cached_entry:
            if cached_entry.depth >= depth:
                self.cache_hits += 1
                cached_best_move = cached_entry.best_move_mask
                if cached_entry.flag == TT_EXACT:
                    return True, alpha, beta, cached_best_move, cached_entry
                elif cached_entry.flag == TT_LOWERBOUND:
                    alpha = max(alpha, cached_entry.score)
                elif cached_entry.flag == TT_UPPERBOUND:
                    beta = min(beta, cached_entry.score)

                if alpha >= beta:
                    return True, alpha, beta, cached_best_move, cached_entry 
                
            elif cached_entry.best_move_mask:
                 cached_best_move = cached_entry.best_move_mask

        else:
            self.cache_misses += 1

        return False, alpha, beta, cached_best_move, cached_entry

    def _store_in_tt(self, board_key: int, score: int, depth: int, flag: int, best_move: int, cached_entry_obj: Optional[TTEntry]):
        "Store to TT"
        if cached_entry_obj is None or depth >= cached_entry_obj.depth:
             entry = TTEntry(score, depth, flag, best_move)
             self.trans_table[board_key] = entry
             self._prune_cache()

    def _negamax(self, board: Board, alpha: int, beta: int, depth: int) -> int:
        "Negamax algorithm"

        self._check_time_limit()
        self.node_count += 1

        win_score = self.SCORE_WINNING_MOVE
        loss_score = -win_score

        opponent_pos_prev = board.current_position ^ board.mask
        if board.has_won(opponent_pos_prev): return loss_score
        if board.nb_moves() >= self.W * self.H: return 0 # Hòa
        winning_moves = board.winning_position() & board.possible()
        if depth <= 0: 
            if winning_moves:
                return win_score
            return self.heuristic(board)

        original_alpha = alpha # store original alpha
        alpha = max(alpha, loss_score)
        beta = min(beta, win_score)
        if alpha >= beta: return alpha

        "Search by TT"
        board_key = board.key()
        can_prune, alpha, beta, tt_best_move, cached_entry = self._handle_tt_lookup(board_key, depth, alpha, beta)
        if can_prune:
             return cached_entry.score

        "Get possible moves & sort them"
        possible = board.possible()
        if possible == 0: 
            return 0
        moves = self._generate_and_sort_moves(board, possible, tt_best_move=tt_best_move)

        best_score = -math.inf
        best_move_found = 0
        move_count = 0

        next_move = moves.getNext()
        while next_move:
            self._check_time_limit()
            move_count += 1
            board_copy = board.copy()
            board_copy.play(next_move)
            score = 0
            
            "PVS"
            if move_count == 1:
                score = -self._negamax(board_copy, -beta, -alpha, depth - 1)
            else:
                "Null windows search"
                self._check_time_limit()
                score = -self._negamax(board_copy, -alpha - 1, -alpha, depth - 1) # Null window
                if score > alpha and score < beta:
                    self._check_time_limit()
                    score = -self._negamax(board_copy, -beta, -alpha, depth - 1) # Re-search

            "Update alpha vs beta"
            if score > best_score:
                best_score = score
                best_move_found = next_move
                if best_score > alpha:
                    alpha = best_score

            "Pruning"
            if alpha >= beta:
                self._store_in_tt(board_key, best_score, depth, TT_LOWERBOUND, best_move_found, cached_entry)
                return best_score # Prune

            next_move = moves.getNext()

        "Store the result to TT"
        final_flag = TT_EXACT if best_score > original_alpha else TT_UPPERBOUND
        move_to_store = best_move_found if final_flag == TT_EXACT else 0
        self._store_in_tt(board_key, best_score, depth, final_flag, move_to_store, cached_entry)

        return best_score

    def get_col_from_move(self, move_mask: Optional[int]) -> int:
        if not move_mask: 
            return -1
        for col in range(self.W):
            if move_mask & self.Board.column_mask(col):
                return col
        return -1