# human_vs_ai.py
"""
人类与 AlphaZero 神经网络 AI 五子棋对战程序

功能:
  - 鼠标点击落子，悬停半透明预览
  - 执黑/执白自由切换
  - 悔棋（撤销人类+AI各一步，支持AI获胜/平局后后悔棋）
  - AI 提示（绿色虚线标记推荐位置）
  - AI 模拟次数动态调节
  - 获胜连线高亮
  - 胜负统计与历史记录

依赖: gamerules.py, network.py, mcts.py, agent_az.py, agent_base.py

用法:
    python human_vs_ai.py
    python human_vs_ai.py --model checkpoints/az_train/best_model.pt --sims 400 --color 1
"""

import tkinter as tk
from tkinter import ttk
import time
import threading
import argparse
from typing import Optional, Tuple, List

from gamerules import GameState, GomokuRules
from agent_az import AZAgent


class HumanVsAIArena:
    """人类 vs AlphaZero AI 可视化对弈竞技场"""

    BS = 15
    CS = 30
    MG = 20
    PR = 12

    def __init__(self, ai_agent: AZAgent, human_color: int = 1):
        self.ai_agent = ai_agent
        self.human_color = human_color

        self.rules = GomokuRules()
        self.state: Optional[GameState] = None

        # ── 统计 ──
        self.wins_human = 0
        self.wins_ai = 0
        self.draws = 0
        self.game_num = 0

        # ── 计时 ──
        self.cur_human_time = 0.0
        self.cur_human_steps = 0
        self.cur_ai_time = 0.0
        self.cur_ai_steps = 0
        self._human_turn_start = 0.0

        # ✅ 每步记录 (is_human, dt)，悔棋时精确回退计时
        self._move_records: List[Tuple[bool, float]] = []
        # ✅ 对局结束时历史文本插入位置，悔棋时精确删除
        self._last_summary_start: Optional[str] = None

        # ── 状态标志 ──
        self.game_running = False
        self.waiting_for_human = False
        self.ai_thinking = False
        self.hover_pos = None

        # ── 启动 ──
        self._build_gui()
        self._start_new_game()
        self.root.mainloop()

    # ════════════════════════════════════════════════════════════
    #  GUI 构建
    # ════════════════════════════════════════════════════════════

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("五子棋 — 人机对弈")
        self.root.resizable(False, False)

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ── 棋盘画布 ──
        cw = 2 * self.MG + (self.BS - 1) * self.CS
        self.canvas = tk.Canvas(
            main, width=cw, height=cw, bg="#DEB887", cursor="hand2"
        )
        self.canvas.pack(side=tk.LEFT, padx=(0, 10))
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_hover)
        self.canvas.bind("<Leave>", self._on_leave)

        # ── 右侧信息面板 ──
        rp = ttk.Frame(main, width=280)
        rp.pack(side=tk.RIGHT, fill=tk.Y)
        rp.pack_propagate(False)

        # 对局信息
        info = ttk.LabelFrame(rp, text="对局信息")
        info.pack(fill=tk.X, pady=(0, 8))

        self.status_var = tk.StringVar(value="准备中...")
        self.turn_var = tk.StringVar(value="")
        self.score_var = tk.StringVar(value="")

        ttk.Label(
            info, textvariable=self.status_var,
            font=("Arial", 11, "bold"), wraplength=250,
        ).pack(anchor="w", padx=5, pady=2)
        ttk.Label(
            info, textvariable=self.turn_var, font=("Arial", 10),
        ).pack(anchor="w", padx=5, pady=2)
        ttk.Label(
            info, textvariable=self.score_var, font=("Arial", 10),
        ).pack(anchor="w", padx=5, pady=2)

        # 用时统计
        tlf = ttk.LabelFrame(rp, text="本局用时统计")
        tlf.pack(fill=tk.X, pady=(0, 8))

        self.human_time_var = tk.StringVar()
        self.ai_time_var = tk.StringVar()

        ttk.Label(
            tlf, textvariable=self.human_time_var, font=("Arial", 10),
        ).pack(anchor="w", padx=5, pady=2)
        ttk.Label(
            tlf, textvariable=self.ai_time_var, font=("Arial", 10),
        ).pack(anchor="w", padx=5, pady=2)

        # 控制区
        ctrl = ttk.LabelFrame(rp, text="控制")
        ctrl.pack(fill=tk.X, pady=(0, 8))

        self.new_btn = ttk.Button(ctrl, text="▶ 新局", command=self._on_new_game_btn)
        self.new_btn.pack(fill=tk.X, padx=5, pady=3)

        btn_row = ttk.Frame(ctrl)
        btn_row.pack(fill=tk.X, padx=5, pady=3)

        self.undo_btn = ttk.Button(btn_row, text="↩ 悔棋", command=self._on_undo)
        self.undo_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        self.hint_btn = ttk.Button(btn_row, text="💡 提示", command=self._on_hint)
        self.hint_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        self.surrender_btn = ttk.Button(
            ctrl, text="🏳 认输", command=self._on_surrender
        )
        self.surrender_btn.pack(fill=tk.X, padx=5, pady=3)

        # 执子选择
        cf = ttk.Frame(ctrl)
        cf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(cf, text="执子:").pack(side=tk.LEFT)
        self.color_var = tk.IntVar(value=self.human_color)
        ttk.Radiobutton(cf, text="黑●(先)", variable=self.color_var, value=1).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Radiobutton(cf, text="白○(后)", variable=self.color_var, value=2).pack(
            side=tk.LEFT, padx=8
        )

        # AI 模拟次数
        sf = ttk.Frame(ctrl)
        sf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(sf, text="AI模拟:").pack(side=tk.LEFT)
        self.sims_var = tk.IntVar(value=self.ai_agent.mcts.num_simulations)
        ttk.Spinbox(
            sf, from_=50, to=2000, increment=50,
            textvariable=self.sims_var, width=6,
        ).pack(side=tk.LEFT, padx=5)

        # 历史记录
        hlf = ttk.LabelFrame(rp, text="历史记录")
        hlf.pack(fill=tk.BOTH, expand=True)

        self.hist_text = tk.Text(
            hlf, width=35, height=12, state="disabled",
            font=("Consolas", 9), bg="#F5F5F5",
        )
        sb = ttk.Scrollbar(hlf, orient="vertical", command=self.hist_text.yview)
        self.hist_text.configure(yscrollcommand=sb.set)
        self.hist_text.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5
        )
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 5))

        # ── 初始绘制 ──
        self._draw_board()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ════════════════════════════════════════════════════════════
    #  坐标转换
    # ════════════════════════════════════════════════════════════

    def _px2bd(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        c = round((px - self.MG) / self.CS)
        r = round((py - self.MG) / self.CS)
        if 0 <= r < self.BS and 0 <= c < self.BS:
            cx = self.MG + c * self.CS
            cy = self.MG + r * self.CS
            if ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 <= self.CS * 0.45:
                return (r, c)
        return None

    def _bd2px(self, r: int, c: int) -> Tuple[int, int]:
        return self.MG + c * self.CS, self.MG + r * self.CS

    # ════════════════════════════════════════════════════════════
    #  鼠标事件
    # ════════════════════════════════════════════════════════════

    def _on_click(self, event):
        if not self.game_running or not self.waiting_for_human:
            return
        pos = self._px2bd(event.x, event.y)
        if pos and self.rules.is_valid_move(self.state, pos):
            dt = time.perf_counter() - self._human_turn_start
            self.waiting_for_human = False
            self._clear_hover()
            self.canvas.delete("hint")
            self._process_move(pos, dt, is_human=True)

    def _on_hover(self, event):
        if not self.game_running or not self.waiting_for_human:
            self._clear_hover()
            return
        pos = self._px2bd(event.x, event.y)
        if pos and self.rules.is_valid_move(self.state, pos):
            if self.hover_pos != pos:
                self._clear_hover()
                self.hover_pos = pos
                r, c = pos
                x, y = self._bd2px(r, c)
                fill = "#444" if self.human_color == 1 else "#EEE"
                self.canvas.create_oval(
                    x - self.PR, y - self.PR, x + self.PR, y + self.PR,
                    fill=fill, outline="gray", stipple="gray50", tags="hover",
                )
        else:
            self._clear_hover()

    def _on_leave(self, event):
        self._clear_hover()

    def _clear_hover(self):
        self.canvas.delete("hover")
        self.hover_pos = None

    # ════════════════════════════════════════════════════════════
    #  绘图
    # ════════════════════════════════════════════════════════════

    def _draw_board(self):
        self.canvas.delete("all")
        for i in range(self.BS):
            p = self.MG + i * self.CS
            self.canvas.create_line(
                self.MG, p, self.MG + (self.BS - 1) * self.CS, p
            )
            self.canvas.create_line(
                p, self.MG, p, self.MG + (self.BS - 1) * self.CS
            )
        for r, c in [(7, 7), (3, 3), (3, 11), (11, 3), (11, 11)]:
            x, y = self._bd2px(r, c)
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black")

    def _draw_piece(self, r, c, player, step_num, last=False):
        x, y = self._bd2px(r, c)
        fill = "black" if player == 1 else "white"
        self.canvas.create_oval(
            x - self.PR, y - self.PR, x + self.PR, y + self.PR,
            fill=fill, outline="black",
        )
        if last:
            self.canvas.create_oval(
                x - self.PR - 1, y - self.PR - 1,
                x + self.PR + 1, y + self.PR + 1,
                outline="red", width=2, tags="last_mark",
            )
        tc = "white" if player == 1 else "black"
        fs = 9 if step_num < 10 else (7 if step_num < 100 else 5)
        self.canvas.create_text(
            x, y, text=str(step_num), fill=tc, font=("Arial", fs, "bold")
        )

    def _draw_winning_line(self, positions: List[Tuple[int, int]]):
        if len(positions) >= 2:
            x0, y0 = self._bd2px(*positions[0])
            x1, y1 = self._bd2px(*positions[-1])
            self.canvas.create_line(
                x0, y0, x1, y1, fill="red", width=3, tags="win_line"
            )
            for r, c in positions:
                x, y = self._bd2px(r, c)
                self.canvas.create_oval(
                    x - self.PR - 3, y - self.PR - 3,
                    x + self.PR + 3, y + self.PR + 3,
                    outline="red", width=2, tags="win_line",
                )

    def _find_winning_line(self, state: GameState) -> List[Tuple[int, int]]:
        if state.last_move is None:
            return []
        r, c = state.last_move
        player = state.board[r * 15 + c]
        if player == 0:
            return []
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            line = [(r, c)]
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < 15 and 0 <= nc < 15 and state.board[nr * 15 + nc] == player:
                    line.append((nr, nc))
                else:
                    break
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < 15 and 0 <= nc < 15 and state.board[nr * 15 + nc] == player:
                    line.insert(0, (nr, nc))
                else:
                    break
            if len(line) >= 5:
                return line
        return []

    def _redraw_all_pieces(self):
        for i, move in enumerate(self.state.history):
            player = 1 if i % 2 == 0 else 2
            self._draw_piece(
                move[0], move[1], player, i + 1,
                last=(i == len(self.state.history) - 1),
            )

    # ════════════════════════════════════════════════════════════
    #  对局流程
    # ════════════════════════════════════════════════════════════

    def _start_new_game(self):
        self.game_num += 1
        self.human_color = self.color_var.get()

        try:
            new_sims = self.sims_var.get()
            if new_sims > 0:
                self.ai_agent.mcts.num_simulations = new_sims
        except (tk.TclError, ValueError):
            pass

        hc = "黑●(先)" if self.human_color == 1 else "白○(后)"
        ac = "白○(后)" if self.human_color == 1 else "黑●(先)"
        self.status_var.set(f"第 {self.game_num} 局  你执{hc}  AI执{ac}")
        self.score_var.set(
            f"总比分  你 {self.wins_human} : AI {self.wins_ai}  平 {self.draws}"
        )

        self.cur_human_time = 0.0
        self.cur_human_steps = 0
        self.cur_ai_time = 0.0
        self.cur_ai_steps = 0
        self._move_records = []
        self._last_summary_start = None

        self.state = GameState(
            board=bytearray(225), current_player=1, history=[], last_move=None
        )
        self._draw_board()

        if hasattr(self.ai_agent, "new_game"):
            self.ai_agent.new_game()

        self.game_running = True
        self.waiting_for_human = False
        self.ai_thinking = False

        self.undo_btn.config(text="↩ 悔棋")
        self.root.after(300, self._do_move)

    def _on_new_game_btn(self):
        self.game_running = False
        self.waiting_for_human = False
        self.ai_thinking = False
        self.root.after(50, self._start_new_game)

    def _on_surrender(self):
        if not self.game_running:
            return
        self._end_game(3 - self.human_color, "人类认输")

    # ════════════════════════════════════════════════════════════
    #  悔棋（核心修改：支持对局结束后悔棋 + 统计回退）
    # ════════════════════════════════════════════════════════════

    def _on_undo(self):
        """悔棋：支持对局中和对局结束后（AI获胜/平局时）撤销

        场景 A — 对局进行中，轮到人类：
          撤销 2 步（人类上一步 + AI 上一步），回到人类重新选择

        场景 B — 对局已结束（AI 获胜 / AI 落子后平局）：
          撤销 2 步（AI 致胜/平局步 + 人类前一步），回退胜负统计，
          删除历史记录中的本局总结，恢复对局

        场景 C — 对局已结束（人类获胜 / 人类落子后平局）：
          撤销 1 步（人类致胜/平局步），回退统计，恢复对局，
          接下来轮到 AI 思考
        """
        # AI 正在思考时不允许悔棋
        if self.ai_thinking:
            return

        moves_to_undo = 0
        rollback_stats = False

        if self.game_running and self.waiting_for_human:
            # ── 场景 A：对局进行中，人类回合 ──
            if len(self.state.history) < 2:
                return
            moves_to_undo = 2

        elif not self.game_running:
            # ── 场景 B / C：对局已结束，允许悔棋 ──
            if len(self.state.history) < 1:
                return

            # 判断最后一手是谁下的
            # history[0] 由黑方(player=1)下，history[1] 由白方(player=2)下，依此类推
            last_mover = 1 if len(self.state.history) % 2 == 1 else 2

            if last_mover != self.human_color:
                # 场景 B：最后一手是 AI 的（AI 获胜或 AI 落子后平局）
                # 需要撤销 AI 的这一步 + 人类的前一步
                if len(self.state.history) < 2:
                    return
                moves_to_undo = 2
            else:
                # 场景 C：最后一手是人类的（人类获胜或人类落子后平局）
                # 只需撤销人类的这一步，回到 AI 回合
                moves_to_undo = 1

            rollback_stats = True
        else:
            return

        # ── 步骤 1：回退胜负统计和历史记录 ──
        if rollback_stats:
            winner = self.rules.check_winner(self.state)
            if winner == 0:
                self.draws = max(0, self.draws - 1)
            elif winner == self.human_color:
                self.wins_human = max(0, self.wins_human - 1)
            else:
                self.wins_ai = max(0, self.wins_ai - 1)

            # 删除历史记录中的最后一条对局总结
            if self._last_summary_start is not None:
                self.hist_text.config(state="normal")
                self.hist_text.delete(self._last_summary_start, tk.END)
                self.hist_text.config(state="disabled")
                self._last_summary_start = None

        # ── 步骤 2：撤销落子并精确回退计时 ──
        for _ in range(moves_to_undo):
            if not self.state.history or not self._move_records:
                break
            move = self.state.history.pop()
            self.state.board[move[0] * 15 + move[1]] = 0
            self.state.current_player = 3 - self.state.current_player

            # 精确回退计时
            is_human, dt = self._move_records.pop()
            if is_human:
                self.cur_human_time = max(0, self.cur_human_time - dt)
                self.cur_human_steps = max(0, self.cur_human_steps - 1)
            else:
                self.cur_ai_time = max(0, self.cur_ai_time - dt)
                self.cur_ai_steps = max(0, self.cur_ai_steps - 1)

        self.state.last_move = self.state.history[-1] if self.state.history else None

        # 重置 AI 搜索树（悔棋后树结构不再有效）
        if hasattr(self.ai_agent, "new_game"):
            self.ai_agent.new_game()

        # ── 步骤 3：重绘棋盘 ──
        self._draw_board()
        self._redraw_all_pieces()
        self._update_time()

        # ── 步骤 4：恢复对局状态 ──
        self.game_running = True
        self.waiting_for_human = False
        self.ai_thinking = False
        self.undo_btn.config(text="↩ 悔棋")

        # 更新比分显示
        self.score_var.set(
            f"总比分  你 {self.wins_human} : AI {self.wins_ai}  平 {self.draws}"
        )

        # 更新状态栏
        hc = "黑●(先)" if self.human_color == 1 else "白○(后)"
        ac = "白○(后)" if self.human_color == 1 else "黑●(先)"
        self.status_var.set(f"第 {self.game_num} 局  你执{hc}  AI执{ac}")

        # 判断轮到谁
        if self.state.current_player == self.human_color:
            self.waiting_for_human = True
            self._human_turn_start = time.perf_counter()
            self.turn_var.set("轮到你落子 (已悔棋)")
        else:
            self.turn_var.set("AI 思考中...")
            self.root.after(100, self._do_move)

    # ════════════════════════════════════════════════════════════
    #  AI 提示
    # ════════════════════════════════════════════════════════════

    def _on_hint(self):
        if not self.game_running or not self.waiting_for_human:
            return

        self.turn_var.set("💡 计算提示中...")

        def worker():
            saved_root = self.ai_agent.mcts.root
            saved_last = self.ai_agent._my_last_action
            move = self.ai_agent.get_move(self.state)
            self.ai_agent.mcts.root = saved_root
            self.ai_agent._my_last_action = saved_last
            if self.game_running and self.waiting_for_human:
                self.root.after(0, self._show_hint, move)

        threading.Thread(target=worker, daemon=True).start()

    def _show_hint(self, move: Tuple[int, int]):
        r, c = move
        x, y = self._bd2px(r, c)
        self.canvas.delete("hint")
        self.canvas.create_oval(
            x - self.PR - 2, y - self.PR - 2,
            x + self.PR + 2, y + self.PR + 2,
            outline="#00CC00", width=3, dash=(6, 3), tags="hint",
        )
        self.canvas.create_text(
            x + self.PR + 8, y - self.PR - 4,
            text="💡", font=("Arial", 10), tags="hint",
        )
        self.turn_var.set(f"💡 提示: ({r},{c}) — 绿色虚线处")
        self.root.after(3000, lambda: self.canvas.delete("hint"))

    # ════════════════════════════════════════════════════════════
    #  用时更新
    # ════════════════════════════════════════════════════════════

    def _update_time(self):
        ah = self.cur_human_time / self.cur_human_steps if self.cur_human_steps else 0
        aa = self.cur_ai_time / self.cur_ai_steps if self.cur_ai_steps else 0
        self.human_time_var.set(f"你:  步数 {self.cur_human_steps}, 平均 {ah:.1f}s")
        self.ai_time_var.set(f"AI:  步数 {self.cur_ai_steps}, 平均 {aa:.3f}s")

    # ════════════════════════════════════════════════════════════
    #  走子与对局控制
    # ════════════════════════════════════════════════════════════

    def _do_move(self):
        if not self.game_running:
            return

        is_human_turn = self.state.current_player == self.human_color

        if is_human_turn:
            self.waiting_for_human = True
            self._human_turn_start = time.perf_counter()
            color_str = "黑●" if self.human_color == 1 else "白○"
            self.turn_var.set(f"轮到你落子 ({color_str}) — 点击棋盘")
        else:
            self.waiting_for_human = False
            self.ai_thinking = True
            self.turn_var.set("AI 思考中...")

            state_ref = self.state

            def worker():
                t0 = time.perf_counter()
                move = self.ai_agent.get_move(state_ref)
                dt = time.perf_counter() - t0
                if self.game_running:
                    self.root.after(0, self._process_move, move, dt, False)

            threading.Thread(target=worker, daemon=True).start()

    def _process_move(self, move, dt, is_human):
        if not self.game_running:
            return

        # 合法性验证（在记录之前检查，确保 _move_records 与 history 同步）
        if not self.rules.is_valid_move(self.state, move):
            if not is_human:
                self.ai_thinking = False
            winner = 3 - self.state.current_player
            self._end_game(winner, "非法落子")
            return

        # 落子
        self.canvas.delete("last_mark")
        self.canvas.delete("hover")
        self.canvas.delete("hint")
        self.rules.apply_move(self.state, move)

        # ✅ 仅在成功落子后记录，保证 _move_records 与 state.history 严格一一对应
        self._move_records.append((is_human, dt))

        if is_human:
            self.cur_human_time += dt
            self.cur_human_steps += 1
        else:
            self.cur_ai_time += dt
            self.cur_ai_steps += 1
            self.ai_thinking = False

        self._update_time()

        player = self.state.board[move[0] * 15 + move[1]]
        step_num = len(self.state.history)
        self._draw_piece(move[0], move[1], player, step_num=step_num, last=True)

        # 胜负判定
        winner = self.rules.check_winner(self.state)
        if winner is not None:
            if winner == 0:
                self._end_game(0, "平局")
            else:
                self._end_game(winner, "五连获胜")
        else:
            self.root.after(50, self._do_move)

    def _end_game(self, winner, reason):
        self.game_running = False
        self.waiting_for_human = False
        self.ai_thinking = False

        if winner == 0:
            self.draws += 1
            result = "平局"
        elif winner == self.human_color:
            self.wins_human += 1
            result = "🎉 你赢了！"
            wl = self._find_winning_line(self.state)
            if wl:
                self._draw_winning_line(wl)
        else:
            self.wins_ai += 1
            result = "AI 获胜"
            wl = self._find_winning_line(self.state)
            if wl:
                self._draw_winning_line(wl)

        ah = self.cur_human_time / self.cur_human_steps if self.cur_human_steps else 0
        aa = self.cur_ai_time / self.cur_ai_steps if self.cur_ai_steps else 0
        ts = self.cur_human_steps + self.cur_ai_steps

        summary = (
            f"=== 第{self.game_num}局 ===\n"
            f"结果: {result} ({reason})\n"
            f"总步数: {ts}\n"
            f"你: {self.cur_human_steps}步 平均{ah:.1f}s\n"
            f"AI: {self.cur_ai_steps}步 平均{aa:.3f}s\n"
            f"总比分: 你{self.wins_human} AI{self.wins_ai} 平{self.draws}\n\n"
        )

        # ✅ 记录插入位置，悔棋时精确删除本条总结
        self._last_summary_start = self.hist_text.index(tk.END)

        self.hist_text.config(state="normal")
        self.hist_text.insert(tk.END, summary)
        self.hist_text.see(tk.END)
        self.hist_text.config(state="disabled")

        self.status_var.set(f"第 {self.game_num} 局结束 — {result}")
        self.score_var.set(
            f"总比分  你 {self.wins_human} : AI {self.wins_ai}  平 {self.draws}"
        )

        # ✅ 对局结束后提示可悔棋
        self.turn_var.set(f"{result}  「新局」开始 / 「悔棋」撤销结果")
        self.undo_btn.config(text="↩ 悔棋(可撤)")

    # ════════════════════════════════════════════════════════════
    #  窗口关闭
    # ════════════════════════════════════════════════════════════

    def _on_closing(self):
        self.game_running = False
        self.waiting_for_human = False
        self.ai_thinking = False
        self.root.destroy()


# ════════════════════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人类 vs AlphaZero 五子棋")
    parser.add_argument(
        "--model", type=str,
        default="checkpoints/az_train/best_model.pt",
        help="模型路径 (默认: checkpoints/az_train/best_model.pt)",
    )
    parser.add_argument(
        "--sims", type=int, default=400,
        help="MCTS 模拟次数 (默认: 400, 越大越强但越慢)",
    )
    parser.add_argument(
        "--color", type=int, default=1, choices=[1, 2],
        help="人类执子: 1=黑(先手), 2=白(后手) (默认: 1)",
    )
    args = parser.parse_args()

    print(f"加载模型: {args.model}")
    print(f"MCTS 模拟次数: {args.sims}")

    ai = AZAgent(
        model_path=args.model,
        num_sims=args.sims,
        temperature=0.0,
        dirichlet_epsilon=0.0,
        name="AlphaZero",
    )

    print("模型加载完成，启动对弈界面...")
    HumanVsAIArena(ai_agent=ai, human_color=args.color)