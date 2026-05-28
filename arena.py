# arena.py
"""
五子棋可视化对弈竞技场
(修复: 移除自导入, 兼容新 gamerules 的 is_board_full, 新增 AZAgent 神经网络对战)
"""

import tkinter as tk
from tkinter import ttk
import time
import threading
from typing import Optional

import torch
import numpy as np

from gamerules import GameState, GomokuRules
from agent_base import Agent
from network import ActorCriticNet
from mcts import MCTS, create_local_eval_fn


# ═══════════════════════════════════════════════════════════════
#  Arena: 可视化对弈竞技场
# ═══════════════════════════════════════════════════════════════

class Arena:
    def __init__(self, agent_black, agent_white):
        self.agent_black = agent_black
        self.agent_white = agent_white
        self.rules = GomokuRules()
        self.state: Optional[GameState] = None

        self.wins_black = 0
        self.wins_white = 0
        self.draws = 0
        self.game_num = 0
        
        self.cur_agent_black_time = 0.0
        self.cur_agent_black_steps = 0
        self.cur_agent_white_time = 0.0
        self.cur_agent_white_steps = 0
        
        self.game_running = False
        self.paused = False
        self.auto_start = True
        self._auto_start_after_id = None

        # 初始化 GUI
        self.root = tk.Tk()
        self.root.title("五子棋无限对弈")
        self.root.resizable(False, False)
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(main_frame, width=450, height=450, bg='#DEB887')
        self.canvas.pack(side=tk.LEFT, padx=(0, 10))

        self.info_frame = ttk.Frame(main_frame, width=280)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_frame.pack_propagate(False)

        # --- 对局信息组 ---
        info_lf = ttk.LabelFrame(self.info_frame, text="对局信息")
        info_lf.pack(fill=tk.X, pady=(0, 10))

        self.status_var = tk.StringVar(value="准备开始...")
        self.score_var = tk.StringVar()
        
        ttk.Label(info_lf, textvariable=self.status_var, font=('Arial', 11, 'bold'), wraplength=250).pack(anchor='w', padx=5, pady=2)
        ttk.Label(info_lf, textvariable=self.score_var, font=('Arial', 10)).pack(anchor='w', padx=5, pady=2)

        # --- 用时统计组 ---
        time_lf = ttk.LabelFrame(self.info_frame, text="本局用时统计")
        time_lf.pack(fill=tk.X, pady=(0, 10))

        self.time_var = tk.StringVar()
        self.black_time_var = tk.StringVar()
        self.white_time_var = tk.StringVar()
        
        ttk.Label(time_lf, textvariable=self.time_var, font=('Arial', 10)).pack(anchor='w', padx=5, pady=2)
        ttk.Label(time_lf, textvariable=self.black_time_var, font=('Arial', 10), foreground='blue').pack(anchor='w', padx=5, pady=2)
        ttk.Label(time_lf, textvariable=self.white_time_var, font=('Arial', 10), foreground='red').pack(anchor='w', padx=5, pady=2)

        # --- 控制组 ---
        ctrl_lf = ttk.LabelFrame(self.info_frame, text="控制")
        ctrl_lf.pack(fill=tk.X, pady=(0, 10))

        self.pause_btn = ttk.Button(ctrl_lf, text="暂停", command=self.toggle_pause)
        self.pause_btn.pack(fill=tk.X, padx=5, pady=5)

        self.auto_start_var = tk.BooleanVar(value=True)
        self.auto_start_cb = ttk.Checkbutton(
            ctrl_lf, text="自动开始新局",
            variable=self.auto_start_var,
            command=self._on_auto_start_toggle
        )
        self.auto_start_cb.pack(fill=tk.X, padx=5, pady=2)

        self.new_game_btn = ttk.Button(
            ctrl_lf, text="▶ 开始新局",
            command=self._manual_start_new_game,
            state='disabled'
        )
        self.new_game_btn.pack(fill=tk.X, padx=5, pady=5)

        # --- 历史记录组 ---
        hist_lf = ttk.LabelFrame(self.info_frame, text="历史记录")
        hist_lf.pack(fill=tk.BOTH, expand=True)

        self.stats_text = tk.Text(hist_lf, width=35, height=15, state='disabled', font=('Consolas', 9), bg='#F5F5F5')
        scrollbar = ttk.Scrollbar(hist_lf, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,0), pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0,5))

        self.draw_board()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.start_new_game)
        self.root.mainloop()

    # ==================== 自动/手动切换 ====================

    def _on_auto_start_toggle(self):
        self.auto_start = self.auto_start_var.get()
        if not self.auto_start:
            if self._auto_start_after_id is not None:
                self.root.after_cancel(self._auto_start_after_id)
                self._auto_start_after_id = None
            if not self.game_running:
                self.new_game_btn.config(state='normal')
        else:
            self.new_game_btn.config(state='disabled')
            if not self.game_running and self._auto_start_after_id is None:
                self.root.after(500, self.start_new_game)

    def _manual_start_new_game(self):
        if not self.game_running:
            if self._auto_start_after_id is not None:
                self.root.after_cancel(self._auto_start_after_id)
                self._auto_start_after_id = None
            self.new_game_btn.config(state='disabled')
            self.start_new_game()

    # ==================== 窗口与暂停 ====================

    def on_closing(self):
        self.game_running = False
        self.paused = False
        if self._auto_start_after_id is not None:
            self.root.after_cancel(self._auto_start_after_id)
            self._auto_start_after_id = None
        self.root.destroy()

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(text="继续")
            current_status = self.status_var.get()
            self.status_var.set(f"[已暂停] {current_status}")
        else:
            self.pause_btn.config(text="暂停")
            if self.game_running:
                self.do_move()

    # ==================== 绘图 ====================

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(15):
            self.canvas.create_line(20, 20 + i*30, 20 + 14*30, 20 + i*30)
            self.canvas.create_line(20 + i*30, 20, 20 + i*30, 20 + 14*30)
        for r,c in [(7,7),(3,3),(3,11),(11,3),(11,11)]:
            x, y = 20 + c*30, 20 + r*30
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill='black')

    def draw_piece(self, r, c, player, step_num, last=False):
        x, y = 20 + c*30, 20 + r*30
        color = 'black' if player == 1 else 'white'
        self.canvas.create_oval(x-12, y-12, x+12, y+12, fill=color, outline='black')
        if last:
            self.canvas.create_oval(x-13, y-13, x+13, y+13, outline='red', width=2, tags="last_mark")
        text_color = 'white' if player == 1 else 'black'
        step_str = str(step_num)
        if step_num < 10: font_size = 9
        elif step_num < 100: font_size = 7
        else: font_size = 5
        self.canvas.create_text(x, y, text=step_str, fill=text_color,
                                font=('Arial', font_size, 'bold'))

    # ==================== 对局流程 ====================

    def _reset_agents(self):
        """新一局开始时重置智能体状态（MCTS 树、增量缓存等）"""
        for agent in [self.agent_black, self.agent_white]:
            if hasattr(agent, 'new_game'):
                agent.new_game()
            elif hasattr(agent, 'reset_incremental_cache'):
                agent.reset_incremental_cache()

    def start_new_game(self):
        self._auto_start_after_id = None
        self.game_num += 1
        self.paused = False
        self.pause_btn.config(text="暂停")
        self.new_game_btn.config(state='disabled')

        if self.game_num % 2 == 1:
            black_agent = self.agent_black
            white_agent = self.agent_white
        else:
            black_agent = self.agent_white
            white_agent = self.agent_black

        black_name = black_agent.name
        white_name = white_agent.name

        self.status_var.set(f"第 {self.game_num} 局  黑: {black_name}  白: {white_name}")
        self.score_var.set(f"总比分 {self.agent_black.name} {self.wins_black} : {self.agent_white.name} {self.wins_white}  平 {self.draws}")
        
        self.cur_agent_black_time = 0.0; self.cur_agent_black_steps = 0
        self.cur_agent_white_time = 0.0; self.cur_agent_white_steps = 0
        self._update_current_game_time()

        self.state = GameState(board=bytearray(225), current_player=1, history=[], last_move=None)
        self.draw_board()

        # ✅ 重置智能体状态（AZAgent 的 MCTS 树、AgentAD 的增量缓存等）
        self._reset_agents()

        self.game_running = True
        self.current_black_agent = black_agent
        self.current_white_agent = white_agent
        self.root.after(500, self.do_move)

    def _update_current_game_time(self):
        avg_black = self.cur_agent_black_time / self.cur_agent_black_steps if self.cur_agent_black_steps else 0
        avg_white = self.cur_agent_white_time / self.cur_agent_white_steps if self.cur_agent_white_steps else 0
        self.black_time_var.set(f"● {self.agent_black.name}: 本局步数 {self.cur_agent_black_steps}, 平均 {avg_black:.3f}s")
        self.white_time_var.set(f"○ {self.agent_white.name}: 本局步数 {self.cur_agent_white_steps}, 平均 {avg_white:.3f}s")

    def do_move(self):
        if not self.game_running or self.paused:
            return

        state = self.state
        agent = self.current_black_agent if state.current_player == 1 else self.current_white_agent

        current_status = self.status_var.get().replace(" 思考中...", "")
        self.status_var.set(f"{current_status} 思考中...")

        def ai_thread_task():
            t0 = time.perf_counter()
            move = agent.get_move(state)
            dt = time.perf_counter() - t0
            
            if self.game_running:
                self.root.after(0, self._process_move, move, dt, agent)

        threading.Thread(target=ai_thread_task, daemon=True).start()

    def _process_move(self, move, dt, agent):
        if not self.game_running:
            return

        if agent == self.agent_black:
            self.cur_agent_black_time += dt; self.cur_agent_black_steps += 1
        else:
            self.cur_agent_white_time += dt; self.cur_agent_white_steps += 1

        total_steps = self.cur_agent_black_steps + self.cur_agent_white_steps
        total_time = self.cur_agent_black_time + self.cur_agent_white_time
        avg_this_game = total_time / total_steps if total_steps else 0
        
        self.time_var.set(f"当前局: 总步数 {total_steps}, 平均 {avg_this_game:.3f}s")
        self._update_current_game_time()

        current_status = self.status_var.get().replace(" 思考中...", "")
        self.status_var.set(current_status)

        if not self.rules.is_valid_move(self.state, move):
            winner = 3 - self.state.current_player
            self.end_game(winner, "非法落子")
            return

        self.canvas.delete("last_mark")
        self.rules.apply_move(self.state, move)

        player = self.state.board[move[0] * 15 + move[1]]
        step_num = len(self.state.history)
        self.draw_piece(move[0], move[1], player, step_num=step_num, last=True)

        winner = self.rules.check_winner(self.state)
        if winner is not None:
            if winner == 0:
                self.end_game(0, "平局")
            else:
                self.end_game(winner, "五连获胜")
        else:
            if not self.paused:
                self.root.after(100, self.do_move)

    def end_game(self, winner, reason):
        self.game_running = False
        self.paused = False
        self.pause_btn.config(text="暂停")

        if winner == 0:
            self.draws += 1
            result = "平局"
            win_color_str = ""
        else:
            win_color_str = "黑" if winner == 1 else "白"
            
            if (self.game_num % 2 == 1 and winner == 1) or (self.game_num % 2 == 0 and winner == 2):
                self.wins_black += 1
                win_name = self.agent_black.name
            else:
                self.wins_white += 1
                win_name = self.agent_white.name
                
            result = f"{win_name}(执{win_color_str}) 胜"

        avg_black = self.cur_agent_black_time / self.cur_agent_black_steps if self.cur_agent_black_steps else 0
        avg_white = self.cur_agent_white_time / self.cur_agent_white_steps if self.cur_agent_white_steps else 0
        total_steps = self.cur_agent_black_steps + self.cur_agent_white_steps
        
        summary = (
            f"=== 第{self.game_num}局结束 ===\n"
            f"结果: {result} ({reason})\n"
            f"总步数: {total_steps}\n"
            f"● {self.agent_black.name}: 本局步数 {self.cur_agent_black_steps}, 平均 {avg_black:.3f}s\n"
            f"○ {self.agent_white.name}: 本局步数 {self.cur_agent_white_steps}, 平均 {avg_white:.3f}s\n"
            f"总比分: {self.agent_black.name} {self.wins_black} : "
            f"{self.agent_white.name} {self.wins_white} (平 {self.draws})\n\n"
        )
        
        self.stats_text.config(state='normal')
        self.stats_text.insert(tk.END, summary)
        self.stats_text.see(tk.END)
        self.stats_text.config(state='disabled')

        self.status_var.set(f"第 {self.game_num} 局结束 - {result}")
        self.score_var.set(f"总比分 {self.agent_black.name} {self.wins_black} : {self.agent_white.name} {self.wins_white}  平 {self.draws}")

        if self.auto_start:
            self._auto_start_after_id = self.root.after(2000, self.start_new_game)
        else:
            self.new_game_btn.config(state='normal')
            self.status_var.set(f"第 {self.game_num} 局结束 - {result}  [点击「开始新局」继续]")


# ═══════════════════════════════════════════════════════════════
#  入口：示例对局配置
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from agent_ad import Agent as ADAgent
    from agent_az import AZAgent
    # ── 普通 AgentAD 对手 ──
    agent1 = ADAgent(
        depth=4, max_candidates=10,
        use_quiescence=True, quiescence_depth=2,
        vct_depth=8, name="ADAgent"
    )

    # ── 神经网络 Agent（参考 az_train._arena_phase 的调用方式） ──
    az_agent1 = AZAgent(
        model_path="checkpoints/joint_pretrain/best_model.pt",
        num_sims=400,
        temperature=0.0,       # 确定性走子（竞技场不探索）
        dirichlet_epsilon=0.0, # 不加 Dirichlet 噪声
        name="AlphaOld",
    )
    az_agent2 = AZAgent(
        model_path="checkpoints/az_train/best_model.pt",
        num_sims=400,
        temperature=0.0,       # 确定性走子（竞技场不探索）
        dirichlet_epsilon=0.0, # 不加 Dirichlet 噪声
        name="AlphaCurr",
    )

    Arena(agent_black=agent1, agent_white=az_agent2)
