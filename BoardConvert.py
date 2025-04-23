import numpy as np
from typing import List, Optional

from board import Board

class GameStateStructure:
    def __init__(self, board_data: List[List[int]], current_player_id: int):
        self.board = board_data
        self.current_player = current_player_id

def create_board_from_gamestate(game_state: GameStateStructure) -> Optional[Board]:

    try:
        board_data = game_state.board
        current_player_id = game_state.current_player

        ROWS = Board.HEIGHT
        COLS = Board.WIDTH

        temp_board = Board() 
        p1_pos = np.int64(0)
        p2_pos = np.int64(0)
        mask = np.int64(0)

        for api_row in range(ROWS):
            for col in range(COLS):
                cell_value = board_data[api_row][col]
                if cell_value != 0:
                    bit_row = (ROWS - 1) - api_row
                    bit_pos = bit_row + col * (Board.HEIGHT + 1)
                    current_bit = np.int64(1) << bit_pos
                    mask |= current_bit 
                    if cell_value == 1:
                        p1_pos |= current_bit
                    elif cell_value == 2:
                        p2_pos |= current_bit
                    else:
                        print(f"Lỗi chuyển đổi: Giá trị ô không hợp lệ ({cell_value}) tại [{api_row}][{col}].")
                        return None

        temp_board.mask = mask
        temp_board.moved_step = Board.pop_count(mask) 

        if current_player_id == 1:
            temp_board.current_position = p1_pos
        elif current_player_id == 2:
            temp_board.current_position = p2_pos

        return temp_board


    except Exception as e:
        print(f"Lỗi không xác định trong quá trình chuyển đổi board: {e}")
        return None

