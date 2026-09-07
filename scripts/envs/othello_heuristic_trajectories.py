"""Trajectory generator for Othello SFT training — heuristic teacher variant.

Heuristic (v2):
  Rule 1 — Corner priority: take any available corner immediately.
  Rule 2 — Corner-state-aware danger filter: an X/C-square is only excluded
            while its associated corner is still unclaimed. Once claimed,
            it's scored via _SAFE_WEIGHT_OUR_CORNER/_SAFE_WEIGHT_OPP_CORNER
            instead of treated as neutral or dangerous.
  Rule 3 — Positional weight + one-ply mobility tie-break: among remaining
            candidates, score = weight + effective_mobility_weight *
            (my_mob - opp_mob), mobility computed via a from-scratch flip
            simulator over the parsed board (not the engine's turn-tracked
            legal_actions(), which is empty for the non-current player).
            effective_mobility_weight scales down as the board fills.

Opponent: in-process MCTS (make_mcts_bot), 25-75 sims, unchanged.

Interface: generate_heuristic_episode(game_id, max_turn) returns
  [(examples_ours, final_reward), (examples_opp, 1 - final_reward)].
"""

import json
import random

from envs.pvp_format import build_full_system_prompt
from envs.pvp_format import build_pvp_tools
from envs.pvp_format import build_user_prompt
from envs.pvp_format import tools_to_openai
from envs.pvp_game_engine import config_id_for_task_id
from envs.pvp_game_engine import make_mcts_bot
from envs.pvp_game_engine import mcts_step_or_none
from envs.pvp_game_engine import OthelloAgent
from envs.pvp_game_engine import score_for_player
from envs.shared_env import _log

_GAME_NAME = "othello"
_AGENT = OthelloAgent()

# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

_ME, _OPP, _EMPTY = 1, -1, 0

# Action IDs: action = row * 8 + col (row 0 = rank 1, col 0 = file a).
_CORNERS   = frozenset({0, 7, 56, 63})          # a1, h1, a8, h8
_X_SQUARES = frozenset({9, 14, 49, 54})          # b2, g2, b7, g7
_C_SQUARES = frozenset({1, 8, 6, 15, 57, 48, 62, 55})  # b1/a2, g1/h2, b8/a7, g8/h7
_PASS_ACTION = 64                                 # OpenSpiel forced-pass token

_CORNER_OF = {
    9: 0, 1: 0, 8: 0,
    14: 7, 6: 7, 15: 7,
    49: 56, 57: 56, 48: 56,
    54: 63, 62: 63, 55: 63,
}

_WEIGHTS = [
    [100, -20,  10,   5,   5,  10, -20, 100],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [ 10,  -2,   5,   1,   1,   5,  -2,  10],
    [  5,  -2,   1,   1,   1,   1,  -2,   5],
    [  5,  -2,   1,   1,   1,   1,  -2,   5],
    [ 10,  -2,   5,   1,   1,   5,  -2,  10],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [100, -20,  10,   5,   5,  10, -20, 100],
]

_SAFE_WEIGHT_OUR_CORNER = 10.0
_SAFE_WEIGHT_OPP_CORNER = 2.0
_MOBILITY_WEIGHT = 3.0
_MOBILITY_PHASE_SCALE = True

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _label(a: int) -> str:
    return chr(ord("a") + a % 8) + str(a // 8 + 1)


def _weight(action_id: int, corner_owner: dict) -> float:
    if action_id in _X_SQUARES or action_id in _C_SQUARES:
        owner = corner_owner.get(_CORNER_OF[action_id])
        if owner == _ME:
            return _SAFE_WEIGHT_OUR_CORNER
        if owner == _OPP:
            return _SAFE_WEIGHT_OPP_CORNER
    return _WEIGHTS[action_id // 8][action_id % 8]


def _parse_board(state_desc: str) -> "list[list[int]] | None":
    """Parse the 8x8 board out of the rendered state text (row 0 = rank 1)."""
    if "You play x" in state_desc:
        me_char, opp_char = "x", "o"
    elif "You play o" in state_desc:
        me_char, opp_char = "o", "x"
    else:
        return None

    rows = []
    for line in state_desc.splitlines():
        parts = line.split()
        if len(parts) == 10 and parts[0] == parts[-1] and parts[0].isdigit():
            cells = parts[1:9]
            if all(c in ("x", "o", "-") for c in cells):
                rows.append(cells)

    if len(rows) != 8:
        return None

    return [[_ME if c == me_char else _OPP if c == opp_char else _EMPTY for c in row] for row in rows]


def _flips(board, r: int, c: int, player: int) -> list:
    if board[r][c] != _EMPTY:
        return []
    opp = -player
    all_flips = []
    for dr, dc in _DIRS8:
        line = []
        nr, nc = r + dr, c + dc
        while 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == opp:
            line.append((nr, nc))
            nr += dr
            nc += dc
        if line and 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == player:
            all_flips.extend(line)
    return all_flips


def _legal_cell_count(board, player: int) -> int:
    return sum(1 for r in range(8) for c in range(8) if _flips(board, r, c, player))


def _apply(board, r: int, c: int, player: int):
    nb = [row[:] for row in board]
    nb[r][c] = player
    for fr, fc in _flips(board, r, c, player):
        nb[fr][fc] = player
    return nb


_TIEBREAK_MARGIN = 0.01  # candidates within this of the top score are considered tied


def _static_eval(board, player: int) -> float:
    """Weight-table score for `player`, ignoring corner-safety/mobility refinements --
    cheap enough to call at every leaf of the depth-2 tie-break lookahead."""
    return sum(
        _WEIGHTS[r][c] for r in range(8) for c in range(8) if board[r][c] == player
    )


def _lookahead_tiebreak(tied: "list[int]", board, corner_owner: dict) -> int:
    """Among heuristic-score ties, pick the move that leaves the opponent the
    weakest best reply two plies out (my move -> opponent's best reply, scored
    by the existing weight table). Only runs on the handful of moves the
    primary heuristic couldn't already separate, so it stays a tie-break, not
    a second teacher -- the label is still "the heuristic's move" in the
    overwhelming majority of turns where there's a unique top score."""
    best_a, best_worst_reply = tied[0], -1e18
    for a in tied:
        r, c = divmod(a, 8)
        nb = _apply(board, r, c, _ME)
        opp_replies = [(rr, cc) for rr in range(8) for cc in range(8) if _flips(nb, rr, cc, _OPP)]
        if not opp_replies:
            # opponent has no reply at all -- as good an outcome as this lookahead can see
            worst_case = 1e18
        else:
            worst_case = min(
                _static_eval(_apply(nb, rr, cc, _OPP), _ME) for rr, cc in opp_replies
            )
        if worst_case > best_worst_reply:
            best_worst_reply = worst_case
            best_a = a
    return best_a


def _heuristic_choose(legal_action_ids: "list[int]", state_desc: str) -> "tuple[int, str]":
    """Apply the heuristic, returning (action_id, reasoning)."""
    if legal_action_ids == [_PASS_ACTION]:
        return _PASS_ACTION, "No board moves available — forced pass."

    board_actions = [a for a in legal_action_ids if a != _PASS_ACTION]

    for a in board_actions:
        if a in _CORNERS:
            return a, f"Corner {_label(a)} is available — taking it."

    board = _parse_board(state_desc)
    if board is None:
        safe = [a for a in board_actions if a not in _X_SQUARES and a not in _C_SQUARES]
        candidates = safe or [a for a in board_actions if a not in _X_SQUARES] or list(board_actions)
        chosen = max(candidates, key=lambda a: _WEIGHTS[a // 8][a % 8])
        return chosen, f"Play {_label(chosen)}."

    corner_owner = {c: board[c // 8][c % 8] for c in _CORNERS if board[c // 8][c % 8] != _EMPTY}
    claimed = set(corner_owner)

    dangerous = {a for a in (_X_SQUARES | _C_SQUARES) if _CORNER_OF[a] not in claimed}
    candidates = [a for a in board_actions if a not in dangerous]
    if not candidates:
        dangerous_x_only = {a for a in _X_SQUARES if _CORNER_OF[a] not in claimed}
        candidates = [a for a in board_actions if a not in dangerous_x_only]
    if not candidates:
        candidates = list(board_actions)

    empty_count = sum(row.count(_EMPTY) for row in board)
    effective_mobility_weight = (
        _MOBILITY_WEIGHT * (empty_count / 64.0) if _MOBILITY_PHASE_SCALE else _MOBILITY_WEIGHT
    )

    scored = {}
    for a in candidates:
        r, c = divmod(a, 8)
        nb = _apply(board, r, c, _ME)
        my_mob = _legal_cell_count(nb, _ME)
        opp_mob = _legal_cell_count(nb, _OPP)
        score = _weight(a, corner_owner) + effective_mobility_weight * (my_mob - opp_mob)
        scored[a] = (score, my_mob, opp_mob)

    best_score = max(scored[a][0] for a in candidates)
    tied = [a for a in candidates if scored[a][0] >= best_score - _TIEBREAK_MARGIN]
    if len(tied) > 1:
        chosen = _lookahead_tiebreak(tied, board, corner_owner)
    else:
        chosen = tied[0]
    _, my_mob, opp_mob = scored[chosen]
    chosen_lbl = _label(chosen)

    parts = []
    skipped_x = [a for a in board_actions if a in dangerous and a in _X_SQUARES]
    skipped_c = [a for a in board_actions if a in dangerous and a in _C_SQUARES and a not in _X_SQUARES]
    if skipped_x:
        parts.append(f"Avoid X-squares ({', '.join(_label(a) for a in skipped_x)}).")
    if skipped_c:
        parts.append(f"Avoid C-squares ({', '.join(_label(a) for a in skipped_c)}).")

    if chosen in dangerous:
        parts.append(f"No safe moves — forced to play {chosen_lbl}.")
    elif chosen in _X_SQUARES or chosen in _C_SQUARES:
        owner = corner_owner.get(_CORNER_OF[chosen])
        corner_lbl = _label(_CORNER_OF[chosen])
        if owner == _ME:
            parts.append(f"Play {chosen_lbl} — extends our corner at {corner_lbl}.")
        else:
            parts.append(f"Play {chosen_lbl} — corner {corner_lbl} is already theirs, no longer risky.")
    else:
        parts.append(f"Play {chosen_lbl} — leaves opponent {opp_mob} replies vs our {my_mob}.")

    return chosen, " ".join(parts)


# ---------------------------------------------------------------------------
# Opponent MCTS config — same range as othello_trajectories.py
# ---------------------------------------------------------------------------

_MCTS_SIMS_MIN = 25
_MCTS_SIMS_MAX = 75

# Toggle: include opponent (lower-sim MCTS) view as additional training examples.
_INCLUDE_OPPONENT_VIEW = False

_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)


def _build_tool_example(
    state_desc: str,
    player_id: int,
    legal_actions: "list[tuple[int, str]]",
    action_id: int,
    reasoning: "str | None" = None,
) -> dict:
    """Build one stateless {system, user, assistant(game_action)} training example."""
    tools = build_pvp_tools([aid for aid, _ in legal_actions])
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state_desc, player_id, legal_actions)},
            {"role": "assistant", "content": reasoning, "tool_calls": [
                {"type": "function", "function": {"name": "game_action", "arguments": json.dumps({"action_id": action_id})}},
            ]},
        ],
        "tools": json.dumps(tools_to_openai(tools)),
    }


def generate_heuristic_episode(
    game_id: int,
    max_turn: int = 70,
) -> "list[tuple[list[dict], float]]":
    """
    Run one Othello game: heuristic teacher vs in-process MCTS opponent.

    Returns [(examples_ours, final_reward), (examples_opp, 1 - final_reward)].
    examples_ours: one per teacher (heuristic) decision, with short reasoning.
    examples_opp: one per MCTS opponent decision (no reasoning, content=None).
    Returns [([], 0.0), ([], 1.0)] on a mid-game error.
    """
    rng = random.Random(game_id)
    teacher_seat = rng.choice((0, 1))
    mcts_simulations = rng.randint(_MCTS_SIMS_MIN, _MCTS_SIMS_MAX)

    game = _AGENT.load_game(_AGENT.generate_params(config_id_for_task_id(game_id)))
    state = game.new_initial_state()
    _AGENT.setup_initial_state(state, seed=game_id)
    opponent_bot = make_mcts_bot(game, mcts_simulations, seed=game_id)

    examples: list[dict] = []
    opp_examples: list[dict] = []

    try:
        for _ in range(max_turn):
            if state.is_terminal():
                break
            cur = state.current_player()
            legal_actions = [(a, state.action_to_string(cur, a)) for a in state.legal_actions(cur)]
            state_desc = _AGENT.format_state(state, cur)

            if cur == teacher_seat:
                action_id, reasoning = _heuristic_choose([aid for aid, _ in legal_actions], state_desc)
                examples.append(_build_tool_example(state_desc, cur, legal_actions, action_id, reasoning))
                state.apply_action(action_id)
            else:
                # Guard: opponent's turn appears here only if it starts first.
                action_id = mcts_step_or_none(opponent_bot, state)
                if action_id is None:
                    _log(f"[othello_heuristic_trajectories] Opponent MCTS failed (game {game_id}), truncating")
                    break
                opp_examples.append(_build_tool_example(state_desc, cur, legal_actions, action_id))
                state.apply_action(action_id)
                continue

            if state.is_terminal():
                break
            opp_cur = state.current_player()
            opp_legal = [(a, state.action_to_string(opp_cur, a)) for a in state.legal_actions(opp_cur)]
            opp_state_desc = _AGENT.format_state(state, opp_cur)
            opp_action_id = mcts_step_or_none(opponent_bot, state)
            if opp_action_id is None:
                _log(f"[othello_heuristic_trajectories] Opponent MCTS failed (game {game_id}), truncating")
                break
            opp_examples.append(_build_tool_example(opp_state_desc, opp_cur, opp_legal, opp_action_id))
            state.apply_action(opp_action_id)
        else:
            _log(f"[othello_heuristic_trajectories] max_turn={max_turn} reached (game {game_id})")
    except Exception as exc:
        _log(f"[othello_heuristic_trajectories] Failed to build episode (game {game_id}): {exc}")
        return [([], 0.0), ([], 1.0)]

    final_reward = score_for_player(state, teacher_seat) if state.is_terminal() else 0.0
    opp_view = (opp_examples, 1.0 - final_reward) if _INCLUDE_OPPONENT_VIEW else ([], 1.0 - final_reward)
    return [(examples, final_reward), opp_view]
