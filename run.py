import time
from board import Board
from solver import Solver

ROWS = 6 
COLS = 7
SEARCH_DEPTH = 11
CACHE_SIZE = 30_000_000
TIME_LIMIT = 8
HUMAN_SYMBOL = 'X'
AI_SYMBOL = 'O'

def _get_fallback_move(game_board: Board, solver: Solver) -> int:
    possible = game_board.possible()
    if possible:
        fallback_move_mask = possible & -possible
        col = solver.get_col_from_move(fallback_move_mask)
        if col != -1:
            return col
    return -1

def get_ai_move(solver: Solver, game_board: Board, depth: int) -> int:
    solver.reset_counters() 

    score, best_move_mask = solver.solve(game_board, depth, time_limit=TIME_LIMIT) # Bỏ time_limit nếu không dùng

    best_col = -1
    if best_move_mask is not None and best_move_mask != 0:
        best_col = solver.get_col_from_move(best_move_mask)

    "if solver failed"
    if best_col == -1:
        best_col = _get_fallback_move(game_board, solver)
        if best_col == -1:
            return -1
    return best_col 

def get_human_move(game_board: Board, player_symbol: str) -> int:
    while True:
        try:
            col_input = input(f"Player {player_symbol}, enter column (1-{COLS}): ")
            col = int(col_input) - 1 # 0-based index

            if not (0 <= col < COLS):
                print(f"Invalid column number. Please enter a number between 1 and {COLS}.")
            elif not game_board.can_play(col):
                print("Column is full. Please choose another column.")
            else:
                return col 

        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
       

def main():
    game_board = Board()
    current_player_is_ai = True
    current_player_symbol = AI_SYMBOL if current_player_is_ai else HUMAN_SYMBOL

    turn_count = 0
    game_over = False
    times = 0
    while not game_over:
        depth = SEARCH_DEPTH
        turn_count += 1
        print(f"\n--- Turn {turn_count} ({current_player_symbol}) ---")
        game_board.print_board() 
        if current_player_is_ai:
            ai_time = time.time()
            solver = Solver(Board, max_cache_size=CACHE_SIZE)
            chosen_col = get_ai_move(solver, game_board, depth)
            if chosen_col == -1:
                print("AI failed to move. Stopping game.")
                game_over = True
                continue 
            ai_end_time = time.time()
            times += (ai_end_time - ai_time)
        else:
            chosen_col = get_human_move(game_board, current_player_symbol)
        game_board.play_col(chosen_col)

        player_who_just_moved_pos = game_board.current_position ^ game_board.mask
        if game_board.has_won(player_who_just_moved_pos):
            print("\n-------------------------")
            game_board.print_board()
            print(f"Player {current_player_symbol} WINS!")
            print("-------------------------")
            game_over = True
        elif game_board.nb_moves() >= ROWS * COLS:
            print("\n-------------------------")
            game_board.print_board()
            print("It's a DRAW!")
            print("-------------------------")
            game_over = True

     
        if not game_over:
            current_player_is_ai = not current_player_is_ai
            current_player_symbol = AI_SYMBOL if current_player_is_ai else HUMAN_SYMBOL
    print(f"Total AI Time: {times}")
    
if __name__ == "__main__":
    main()