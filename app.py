from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

from board import Board
from solver import Solver
import numpy as np
from BoardConvert import create_board_from_gamestate, GameStateStructure


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

@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
            
        SEARCH_DEPTH = 11
        CACHE_SIZE = 10_000_000
        TIME_LIMIT = 8
        NUM_MID_MOVE = 1
        ai_chose_col = -1
        

        solver = Solver(Board, max_cache_size=CACHE_SIZE)

        temp_state_struct = GameStateStructure(
            board_data=game_state.board,
            current_player_id=game_state.current_player
        )
        board_instance = create_board_from_gamestate(temp_state_struct)
        
        center_col = Board.WIDTH // 2

        if board_instance.nb_moves() < NUM_MID_MOVE * 2:
            ai_chose_col = center_col
            
        if ai_chose_col == -1:
            solver.reset_counters()
            score, best_move_mask = solver.solve(board_instance, SEARCH_DEPTH, time_limit=TIME_LIMIT)

            if best_move_mask is not None and best_move_mask != 0:
                col_result = solver.get_col_from_move(best_move_mask)
                if col_result != -1 and col_result in game_state.valid_moves:
                    ai_chose_col = col_result
                else:
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


# https://dashboard.render.com/web/srv-d03g2rjuibrs73a8aao0/logs