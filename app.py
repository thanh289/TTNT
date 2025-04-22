from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

from board import Board
from solver import Solver
import numpy as np


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]
    is_new_game: bool

class AIResponse(BaseModel):
    move: int

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
            
        ROWS = 6
        COLS = 7
        SEARCH_DEPTH = 11
        CACHE_SIZE = 10_000_000
        TIME_LIMIT = 8
        ai_chose_col = -1
        board_instance = None

        try:
            solver = Solver(Board, max_cache_size=CACHE_SIZE)
            if game_state.is_new_game:
                solver.clear_cache()

            temp_board = Board()
            p1_pos = np.int64(0)
            p2_pos = np.int64(0)
            mask = np.int64(0)
            num_rows_api = len(game_state.board)
            num_cols_api = len(game_state.board[0]) if num_rows_api > 0 else 0

            if num_rows_api == ROWS and num_cols_api == COLS:
                for api_row in range(ROWS):
                    for col in range(COLS):
                        cell_value = game_state.board[api_row][col]
                        if cell_value != 0:
                            bit_row = (ROWS - 1) - api_row
                            bit_pos = bit_row + col * (Board.HEIGHT + 1)
                            current_bit = np.int64(1) << bit_pos
                            mask |= current_bit
                            if cell_value == 1: p1_pos |= current_bit
                            elif cell_value == 2: p2_pos |= current_bit
                temp_board.mask = mask
                temp_board.moved_step = Board.pop_count(mask)
                if game_state.current_player == 1:
                    temp_board.current_position = p1_pos
                elif game_state.current_player == 2:
                    temp_board.current_position = p2_pos
                else: temp_board = None
                if temp_board and (temp_board.current_position & temp_board.mask) == temp_board.current_position:
                    board_instance = temp_board
            
            if board_instance:
                solver.reset_counters()
                score, best_move_mask = solver.solve(board_instance, SEARCH_DEPTH, time_limit=TIME_LIMIT)
                if best_move_mask is not None and best_move_mask != 0:
                    col_result = solver.get_col_from_move(best_move_mask)
                    if col_result != -1 and col_result in game_state.valid_moves:
                        ai_chose_col = col_result

        except Exception as e_inner:
             print(f"AI logic error: {e_inner}") # Vẫn nên giữ lại log lỗi này
             ai_chose_col = -1

        if ai_chose_col == -1:
            selected_move = random.choice(game_state.valid_moves)
        else:
            selected_move = ai_chose_col

        
        return AIResponse(move=selected_move)
    except Exception as e:
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)