import tkinter as tk
from tkinter import messagebox

class GoBoardGUI:
    """
    tkinter를 사용하여 19x19 바둑판 GUI를 구현하는 클래스입니다.
    """
    def __init__(self, master):
        self.master = master
        self.master.title("파이썬 GUI 바둑판")
        self.master.resizable(False, False) # 창 크기 조절 비활성화

        self.BOARD_SIZE = 19  # 바둑판 크기 (19x19)
        self.CELL_SIZE = 30   # 각 칸의 크기 (픽셀 단위)
        self.STONE_RADIUS = 12 # 돌의 반지름 (픽셀 단위)
        self.BORDER_OFFSET = self.CELL_SIZE # 바둑판 테두리 여백 (좌표 계산의 기준점)

        # 캔버스 크기 계산 (바둑판 크기 + 양쪽 여백)
        self.CANVAS_WIDTH = self.BOARD_SIZE * self.CELL_SIZE + 2 * self.BORDER_OFFSET
        self.CANVAS_HEIGHT = self.BOARD_SIZE * self.CELL_SIZE + 2 * self.BORDER_OFFSET

        # 캔버스 생성 및 배치
        # 배경색은 바둑판에 흔히 사용되는 나무색 계열로 설정했습니다.
        self.canvas = tk.Canvas(master, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg="#DDCB97")
        self.canvas.pack(padx=10, pady=10)

        # 바둑판 논리적 상태 초기화 (0: 빈 칸, 1: 흑돌, 2: 백돌)
        self.board = [[0 for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.current_player = 1 # 1: 흑돌, 2: 백돌 (흑돌부터 시작)

        # 초기 바둑판 그리기
        self.draw_board()
        # 마우스 왼쪽 클릭 이벤트를 handle_click 메서드에 바인딩
        self.canvas.bind("<Button-1>", self.handle_click)

        # 게임 상태를 표시하는 라벨
        self.status_label = tk.Label(master, text="현재 차례: 흑돌", font=("맑은 고딕", 12, "bold"))
        self.status_label.pack(pady=5)

    def draw_board(self):
        """
        바둑판의 선, 화점, 그리고 놓여진 돌들을 그립니다.
        매 턴마다 캔버스를 지우고 다시 그리는 방식으로 상태를 업데이트합니다.
        """
        self.canvas.delete("all") # 캔버스에 그려진 모든 객체 삭제

        # 바둑판 선 그리기
        for i in range(self.BOARD_SIZE):
            # 가로선 그리기
            self.canvas.create_line(self.BORDER_OFFSET, 
                                    self.BORDER_OFFSET + i * self.CELL_SIZE,
                                    self.CANVAS_WIDTH - self.BORDER_OFFSET, 
                                    self.BORDER_OFFSET + i * self.CELL_SIZE,
                                    fill="black", width=1)
            # 세로선 그리기
            self.canvas.create_line(self.BORDER_OFFSET + i * self.CELL_SIZE,
                                    self.BORDER_OFFSET,
                                    self.BORDER_OFFSET + i * self.CELL_SIZE,
                                    self.CANVAS_HEIGHT - self.BORDER_OFFSET,
                                    fill="black", width=1)
        
        # 화점(Hoshi) 그리기
        # 바둑판의 주요 기준점들을 표시합니다. (일반적으로 3-3, 9-9 등)
        hoshi_indices = [3, 9, 15] # 0-indexed 기준
        for r in hoshi_indices:
            for c in hoshi_indices:
                x_pixel, y_pixel = self._get_pixel_coords(r, c)
                # 화점은 작은 검은색 원으로 표시
                self.canvas.create_oval(x_pixel - 3, y_pixel - 3, x_pixel + 3, y_pixel + 3,
                                        fill="black", outline="black")

        self.draw_stones() # 현재 board 상태에 따라 돌들을 그립니다.

    def draw_stones(self):
        """
        현재 바둑판(self.board) 상태에 따라 모든 돌을 캔버스에 그립니다.
        """
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                stone_color_code = self.board[r][c]
                if stone_color_code != 0: # 돌이 놓여있는 칸인 경우
                    x_pixel, y_pixel = self._get_pixel_coords(r, c)
                    # 돌 색상과 테두리 색상 설정
                    color = "black" if stone_color_code == 1 else "white"
                    outline_color = "white" if stone_color_code == 1 else "black" 
                    
                    self.canvas.create_oval(x_pixel - self.STONE_RADIUS, y_pixel - self.STONE_RADIUS,
                                            x_pixel + self.STONE_RADIUS, y_pixel + self.STONE_RADIUS,
                                            fill=color, outline=outline_color, width=1)

    def handle_click(self, event):
        """
        마우스 클릭 이벤트를 처리하여 사용자가 돌을 놓는 동작을 수행합니다.
        """
        # 클릭된 픽셀 좌표를 바둑판 그리드 좌표 (행, 열)로 변환
        row, col = self._get_board_coords(event.x, event.y)

        # 변환된 좌표가 유효한 바둑판 범위 내에 있는지 확인
        if 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE:
            # 해당 위치에 이미 돌이 놓여있는지 확인
            if self.board[row][col] == 0: # 빈 칸인 경우에만 돌을 놓을 수 있음
                self.board[row][col] = self.current_player # 현재 플레이어의 돌 놓기
                self.draw_stones() # 돌이 새로 놓였으므로 캔버스에 다시 그리기

                # 다음 플레이어로 턴 전환 (1 -> 2, 2 -> 1)
                self.current_player = 3 - self.current_player 
                self.update_status_label() # 상태 라벨 업데이트
            else:
                # 이미 돌이 놓여진 경우 경고 메시지 출력
                messagebox.showwarning("경고", "이미 돌이 놓여진 자리입니다!")
        # else: 바둑판 영역 밖 클릭은 무시

    def _get_board_coords(self, pixel_x, pixel_y):
        """
        캔버스 픽셀 좌표 (pixel_x, pixel_y)를 바둑판 그리드 좌표 (행, 열)로 변환합니다.
        가장 가까운 교차점을 찾아 반환합니다.
        """
        # 보더 오프셋을 제외한 후 셀 크기로 나누어 인덱스를 얻고, 반올림하여 가장 가까운 교차점을 찾습니다.
        col = round((pixel_x - self.BORDER_OFFSET) / self.CELL_SIZE)
        row = round((pixel_y - self.BORDER_OFFSET) / self.CELL_SIZE)
        return row, col

    def _get_pixel_coords(self, row, col):
        """
        바둑판 그리드 좌표 (row, col)를 캔버스 픽셀 좌표 (x_pixel, y_pixel)로 변환합니다.
        각 교차점의 중심 좌표를 반환합니다.
        """
        x_pixel = self.BORDER_OFFSET + col * self.CELL_SIZE
        y_pixel = self.BORDER_OFFSET + row * self.CELL_SIZE
        return x_pixel, y_pixel

    def update_status_label(self):
        """
        현재 플레이어 정보를 상태 라벨에 업데이트하여 사용자에게 보여줍니다.
        """
        player_text = "흑돌" if self.current_player == 1 else "백돌"
        self.status_label.config(text=f"현재 차례: {player_text}")

# 메인 애플리케이션 실행 부분
if __name__ == "__main__":
    root = tk.Tk() # Tkinter 윈도우 객체 생성
    game = GoBoardGUI(root) # GoBoardGUI 클래스 인스턴스 생성
    root.mainloop() # Tkinter 이벤트 루프 시작 (GUI를 계속 실행)