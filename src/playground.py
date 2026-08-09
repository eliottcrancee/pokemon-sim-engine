from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any

from tqdm import tqdm

from pokemon.agents import (
    AlphaBetaAgent,
    BaseAgent,
    BestAttackAgent,
    BestAttackAndPotionAgent,
    InputAgent,
    IterativeDeepeningAlphaBetaAgent,
    MCTSAgent,
    MinimaxAgent,
    MonteCarloExpectiminimaxAgent,
    OneStepUniformExpectimaxAgent,
    RandomAgent,
    RandomAttackAgent,
    RandomAttackAndPotionAgent,
    RiskAwareExpectimaxAgent,
    SmarterHeuristicAgent,
)
from pokemon.battle import Battle
from pokemon.battle_registry import BattleRegistry
from pokemon.play import play_multiple, play_tournament
from pokemon.ui import play_ui


@dataclass(frozen=True, slots=True)
class AgentEntry:
    key: str
    label: str
    factory: Callable[[], BaseAgent]


@dataclass(frozen=True, slots=True)
class MenuAction:
    key: str
    label: str
    handler: Callable[[], None]


AGENT_CATALOG: list[AgentEntry] = [
    AgentEntry("random", "Random (all actions)", lambda: RandomAgent()),
    AgentEntry("random_attack", "Random Attack", lambda: RandomAttackAgent()),
    AgentEntry("best_attack", "Best Attack", lambda: BestAttackAgent()),
    AgentEntry(
        "best_attack_potion",
        "Best Attack + Potion",
        lambda: BestAttackAndPotionAgent(),
    ),
    AgentEntry(
        "smarter_heuristic",
        "Smarter Heuristic",
        lambda: SmarterHeuristicAgent(),
    ),
    AgentEntry(
        "uniform_expectimax",
        "One-Step Uniform Expectimax",
        lambda: OneStepUniformExpectimaxAgent(),
    ),
    AgentEntry(
        "minimax_d2",
        "Minimax d=2",
        lambda: MinimaxAgent(depth=2),
    ),
    AgentEntry(
        "alphabeta_d2",
        "Alpha-Beta d=2",
        lambda: AlphaBetaAgent(depth=2),
    ),
    AgentEntry(
        "alphabeta_d4_parallel",
        "Alpha-Beta d=4 (parallel)",
        lambda: AlphaBetaAgent(depth=4, parallelize=True),
    ),
    AgentEntry(
        "risk_expectimax",
        "Risk-Aware Expectimax",
        lambda: RiskAwareExpectimaxAgent(depth=1),
    ),
    AgentEntry(
        "mc_expectiminimax",
        "Monte Carlo Expectiminimax",
        lambda: MonteCarloExpectiminimaxAgent(depth=1),
    ),
    AgentEntry(
        "iterative_alphabeta_short",
        "Iterative Deepening Alpha-Beta (short time limit)",
        lambda: IterativeDeepeningAlphaBetaAgent(max_depth=5, time_limit=0.5),
    ),
    AgentEntry(
        "iterative_alphabeta_long",
        "Iterative Deepening Alpha-Beta (long time limit)",
        lambda: IterativeDeepeningAlphaBetaAgent(max_depth=5, time_limit=2.0),
    ),
    AgentEntry("mcts", "MCTS", lambda: MCTSAgent(time_limit=1.0, rollout_depth=8)),
]

# Agents used as fixed baselines in sweep mode.
BASELINE_AGENT_KEYS: tuple[str, ...] = (
    "random",
    "random_attack",
    "best_attack",
    "best_attack_potion",
    "smarter_heuristic",
    "uniform_expectimax",
    "alphabeta_d2",
    "risk_expectimax",
)

SWEEP_FAMILIES: list[type[BaseAgent]] = [
    AlphaBetaAgent,
    MinimaxAgent,
    IterativeDeepeningAlphaBetaAgent,
    MCTSAgent,
    MonteCarloExpectiminimaxAgent,
    RiskAwareExpectimaxAgent,
    SmarterHeuristicAgent,
    BestAttackAndPotionAgent,
    RandomAttackAndPotionAgent,
]


def _family_display_name(agent_cls: type[BaseAgent]) -> str:
    explicit = getattr(agent_cls, "SWEEP_DISPLAY_NAME", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    raw = agent_cls.__name__.replace("Agent", "")
    tokens: list[str] = []
    current = ""
    for char in raw:
        if char.isupper() and current:
            tokens.append(current)
            current = char
        else:
            current += char
    if current:
        tokens.append(current)
    return " ".join(tokens)


def _agent_entry_map() -> dict[str, AgentEntry]:
    return {entry.key: entry for entry in AGENT_CATALOG}


def _safe_input(prompt: str) -> str:
    return input(prompt).strip()


def _prompt_int(prompt: str, minimum: int = 1, default: int | None = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = _safe_input(f"{prompt}{suffix}: ")
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
            if value < minimum:
                print(f"Value must be >= {minimum}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer.")


def _prompt_optional_int(prompt: str) -> int | None:
    while True:
        raw = _safe_input(f"{prompt} [empty=random]: ")
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer or empty.")


def _select_from_list(prompt: str, options: list[str]) -> int:
    print(f"\n{prompt}")
    for index, option in enumerate(options, 1):
        print(f"  [{index}] {option}")

    while True:
        raw = _safe_input("> ")
        try:
            selected = int(raw)
            if 1 <= selected <= len(options):
                return selected - 1
            print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Please enter a valid number.")


def _build_battle_description(battle: Battle) -> str:
    def describe_trainer(trainer) -> str:
        team = ", ".join(f"{p.name} L{p.level}" for p in trainer.pokemon_team)
        return f"{trainer.name}: {team}"

    return (
        f"{describe_trainer(battle.trainers[0])} | "
        f"{describe_trainer(battle.trainers[1])}"
    )


def _select_battle(prompt_seed: bool = True) -> tuple[str, Battle]:
    battle_names = BattleRegistry.list_battles()
    if not battle_names:
        raise RuntimeError("Battle registry is empty.")

    labels: list[str] = []
    for name in battle_names:
        battle = BattleRegistry.get(name)
        if battle is None:
            raise RuntimeError(f"Battle '{name}' not found.")
        labels.append(f"{name} -> {_build_battle_description(battle)}")

    selected_index = _select_from_list("Choose a battle scenario:", labels)
    selected_name = battle_names[selected_index]
    battle = BattleRegistry.get(selected_name)
    if battle is None:
        raise RuntimeError(f"Battle '{selected_name}' not found.")

    if prompt_seed:
        requested_seed = _prompt_optional_int("Seed")
        if requested_seed is not None:
            battle.reseed(requested_seed)
            print(f"Seed fixed to {battle.seed}.")
        else:
            print(f"Using battle seed {battle.seed}.")
    else:
        print("Seed prompt skipped: this mode reseeds each battle automatically.")

    return selected_name, battle


def _select_agent(prompt: str) -> AgentEntry:
    labels = [entry.label for entry in AGENT_CATALOG]
    selected_index = _select_from_list(prompt, labels)
    return AGENT_CATALOG[selected_index]


def _parse_csv_values(raw: str, kind: str, default_values: list[Any]) -> list[Any]:
    text = raw.strip()
    if not text:
        return default_values

    values = [part.strip() for part in text.split(",") if part.strip()]
    if kind == "bool":
        truthy = {"1", "true", "t", "yes", "y", "on"}
        falsy = {"0", "false", "f", "no", "n", "off"}
        parsed: list[bool] = []
        for token in values:
            lowered = token.lower()
            if lowered in truthy:
                parsed.append(True)
            elif lowered in falsy:
                parsed.append(False)
            else:
                raise ValueError(f"Invalid boolean token: {token}")
        return parsed

    if kind == "int":
        parsed_ints = [int(token) for token in values]
        if any(v <= 0 for v in parsed_ints):
            raise ValueError("Integer values must be > 0")
        return parsed_ints

    if kind == "float":
        parsed_floats = [float(token) for token in values]
        return parsed_floats

    raise ValueError(f"Unsupported parameter type: {kind}")


def _infer_kind(values: list[Any]) -> str:
    if not values:
        raise ValueError("Default sweep parameter values cannot be empty.")
    sample = values[0]
    if isinstance(sample, bool):
        return "bool"
    if isinstance(sample, int):
        return "int"
    if isinstance(sample, float):
        return "float"
    raise ValueError(f"Unsupported sweep parameter type: {type(sample).__name__}")


def _generate_sweep_agents(agent_cls: type[BaseAgent]) -> list[BaseAgent]:
    defaults = getattr(agent_cls, "DEFAULT_SWEEP_PARAMS", None)
    if not isinstance(defaults, dict) or not defaults:
        raise ValueError(f"{agent_cls.__name__} has no DEFAULT_SWEEP_PARAMS")

    keys: list[str] = []
    value_lists: list[list[Any]] = []

    for param_name, default_values in defaults.items():
        kind = _infer_kind(default_values)
        default_text = ",".join(str(v) for v in default_values)
        raw = _safe_input(f"{param_name} values (default {default_text}): ")
        values = _parse_csv_values(raw, kind, default_values)
        keys.append(param_name)
        value_lists.append(values)

    combinations = [dict(zip(keys, combo)) for combo in product(*value_lists)]
    return [agent_cls(**kwargs) for kwargs in combinations]


def _print_battle_results(
    agent_left: BaseAgent,
    agent_right: BaseAgent,
    wins_left: int,
    draws: int,
    wins_right: int,
    n_battles: int,
) -> None:
    print("\nResults")
    print(f"- {agent_left.name}: {wins_left} ({wins_left / n_battles:.1%})")
    print(f"- {agent_right.name}: {wins_right} ({wins_right / n_battles:.1%})")
    print(f"- Draws: {draws} ({draws / n_battles:.1%})")


def _print_rankings(agents: list[BaseAgent], title: str) -> None:
    if not agents:
        print("No ranking to display.")
        return

    max_name = max(len(a.name) for a in agents)
    print(f"\n{title}")
    header = f"{'Rank':<5} {'Agent':<{max_name}} {'Rating':>8} {'RD':>8}"
    print(header)
    print("-" * len(header))
    for index, agent in enumerate(agents, 1):
        print(
            f"{index:<5} {agent.name:<{max_name}} "
            f"{agent.rating:>8.2f} {agent.rating_deviation:>8.2f}"
        )


def run_ui_duel() -> None:
    print("\n=== UI Duel ===")
    opponent_entry = _select_agent("Choose the AI opponent:")
    battle_name, battle = _select_battle()

    print(f"Launching UI duel on '{battle_name}' against {opponent_entry.label}...")
    time.sleep(0.6)

    player = InputAgent()
    opponent = opponent_entry.factory()
    play_ui(battle, player, opponent)


def run_agent_vs_agent() -> None:
    print("\n=== Agent vs Agent ===")
    left_entry = _select_agent("Choose Agent 1:")
    right_entry = _select_agent("Choose Agent 2:")
    battle_name, battle = _select_battle(prompt_seed=False)
    n_battles = _prompt_int("Number of battles", minimum=1, default=50)

    left = left_entry.factory()
    right = right_entry.factory()

    print(
        f"Running {n_battles} battles on '{battle_name}' -> {left.name} vs {right.name}"
    )

    wins_left, draws, wins_right = play_multiple(
        battle=battle,
        agent_0=left,
        agent_1=right,
        n_battles=n_battles,
        n_jobs=-1,
        verbose=True,
        reseed_each_battle=True,
    )

    _print_battle_results(left, right, wins_left, draws, wins_right, n_battles)


def run_tournament() -> None:
    print("\n=== Tournament (Glicko-2) ===")
    battle_name, battle = _select_battle(prompt_seed=False)
    n_matches = _prompt_int("Total matches", minimum=1, default=150)
    n_battles_per_match = _prompt_int("Battles per match", minimum=1, default=5)

    entry_map = _agent_entry_map()
    pool: list[BaseAgent] = [entry_map[key].factory() for key in BASELINE_AGENT_KEYS]

    print(
        f"Running tournament on '{battle_name}' with {len(pool)} agents, "
        f"{n_matches} matches, {n_battles_per_match} battles per match."
    )

    rankings = play_tournament(
        agent_pool=pool,
        battle=battle,
        n_matches=n_matches,
        n_battles_per_match=n_battles_per_match,
        reseed_each_battle=True,
    )
    _print_rankings(rankings, "Final rankings")


def run_hyperparameter_sweep() -> None:
    print("\n=== Hyperparameter Sweep ===")
    battle_name, battle = _select_battle(prompt_seed=False)

    family_labels = [_family_display_name(cls) for cls in SWEEP_FAMILIES]
    family_index = _select_from_list("Choose an agent family to sweep:", family_labels)
    family_cls = SWEEP_FAMILIES[family_index]
    family_label = family_labels[family_index]

    variants = _generate_sweep_agents(family_cls)
    if not variants:
        print("No variants generated.")
        return

    n_matches = _prompt_int("Total matches", minimum=1, default=200)
    n_battles_per_match = _prompt_int("Battles per match", minimum=1, default=5)

    entry_map = _agent_entry_map()
    baseline_agents: list[BaseAgent] = [
        entry_map[key].factory() for key in BASELINE_AGENT_KEYS
    ]

    print(
        f"Running sweep for '{family_label}' on '{battle_name}' "
        f"with {len(variants)} variants."
    )

    full_pool = baseline_agents + variants
    rankings = play_tournament(
        agent_pool=full_pool,
        battle=battle,
        n_matches=n_matches,
        n_battles_per_match=n_battles_per_match,
        reseed_each_battle=True,
    )

    variant_names = {agent.name for agent in variants}
    variant_rankings = [agent for agent in rankings if agent.name in variant_names]

    _print_rankings(variant_rankings, f"Sweep rankings: {family_label}")


def run_agent_benchmark() -> None:
    print("\n=== Agent Action Benchmark ===")
    entry = _select_agent("Choose an agent to benchmark:")
    opponent_entry = _select_agent("Choose opponent agent:")
    battle_name, battle_template = _select_battle(prompt_seed=False)
    repetitions = _prompt_int("Number of full battles", minimum=1, default=100)

    agent = entry.factory()
    opponent = opponent_entry.factory()

    battle_durations: list[float] = []
    rounds_per_battle: list[int] = []
    agent_call_durations: list[float] = []

    progress = tqdm(range(repetitions), desc="Benchmark")
    for index in progress:
        battle_instance = battle_template.copy()
        battle_instance.reset(reseed=True)

        battle_start = time.perf_counter()
        while not battle_instance.done:
            call_start = time.perf_counter()
            action_0 = agent.get_action(battle_instance, trainer_id=0, verbose=False)
            agent_call_durations.append(time.perf_counter() - call_start)

            action_1 = opponent.get_action(battle_instance, trainer_id=1, verbose=False)
            battle_instance.turn(action_0, action_1)
        battle_durations.append(time.perf_counter() - battle_start)
        rounds_per_battle.append(battle_instance.round)

    min_battle_time = min(battle_durations)
    max_battle_time = max(battle_durations)
    mean_battle_time = statistics.mean(battle_durations)
    std_battle_time = (
        statistics.pstdev(battle_durations) if len(battle_durations) > 1 else 0.0
    )

    total_rounds = sum(rounds_per_battle)
    mean_rounds = statistics.mean(rounds_per_battle)
    mean_battle_time_per_round = (
        (sum(battle_durations) / total_rounds) if total_rounds > 0 else 0.0
    )

    min_agent_call = min(agent_call_durations)
    max_agent_call = max(agent_call_durations)
    mean_agent_call = statistics.mean(agent_call_durations)
    std_agent_call = (
        statistics.pstdev(agent_call_durations)
        if len(agent_call_durations) > 1
        else 0.0
    )

    print("\nBenchmark summary")
    print(f"- Agent: {entry.label}")
    print(f"- Opponent: {opponent_entry.label}")
    print(f"- Battle: {battle_name}")
    print(f"- Full battles: {repetitions}")
    print(f"- Mean rounds per battle: {mean_rounds:.2f}")
    print(f"- Total rounds: {total_rounds}")
    print("\nFull battle timings")
    print(f"- Min: {min_battle_time:.6f}s")
    print(f"- Max: {max_battle_time:.6f}s")
    print(f"- Mean: {mean_battle_time:.6f}s")
    print(f"- Std Dev: {std_battle_time:.6f}s")
    print(f"- Mean battle time / round: {mean_battle_time_per_round:.6f}s")
    print("\nAgent get_action timings (trainer 0 only)")
    print(f"- Calls measured: {len(agent_call_durations)}")
    print(f"- Min: {min_agent_call:.6f}s")
    print(f"- Max: {max_agent_call:.6f}s")
    print(f"- Mean: {mean_agent_call:.6f}s")
    print(f"- Std Dev: {std_agent_call:.6f}s")


def main() -> None:
    menu_actions = [
        MenuAction("ui", "UI duel (human vs AI)", run_ui_duel),
        MenuAction("sim", "Agent vs Agent simulation", run_agent_vs_agent),
        MenuAction("tour", "Tournament (Glicko-2)", run_tournament),
        MenuAction("sweep", "Hyperparameter sweep", run_hyperparameter_sweep),
        MenuAction("bench", "Agent action benchmark", run_agent_benchmark),
        MenuAction("quit", "Exit", lambda: None),
    ]

    while True:
        options = [action.label for action in menu_actions]
        selected_index = _select_from_list("Pokemon Playground - main menu", options)
        selected_action = menu_actions[selected_index]

        if selected_action.key == "quit":
            print("Exiting playground.")
            return

        try:
            selected_action.handler()
        except Exception as error:
            print(f"\nError: {error}")

        _safe_input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()
