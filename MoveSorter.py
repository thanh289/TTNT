import numpy as np

class MoveSorter:
    """
    Sắp xếp các nước đi theo điểm số, tương tự priority queue
    để lấy ra nước đi có giá trị tốt nhất.
    """
    
    def __init__(self, board_width):
        self.size = 0
        self.board_width = board_width
        self.entries = [{'move': 0, 'score': 0} for _ in range(board_width)]
    
    def add(self, move, score):
        "Thêm một nước đi vào entries với điểm số của nó"
        move = int(move)
        pos = self.size
        self.size += 1
        
        "insertion sort"
        while pos > 0 and self.entries[pos-1]['score'] > score:
            self.entries[pos] = self.entries[pos-1].copy()
            pos -= 1
        
        self.entries[pos] = {'move': move, 'score': score}
    

    def getNext(self):
        "Lấy và return nước đi tiếp theo (có điểm số cao nhất) hoặc return 0 nếu không còn nước đi nào"
        if self.size > 0:
            self.size -= 1
            return self.entries[self.size]['move']
        else:
            return 0
    
    def reset(self):
        self.size = 0