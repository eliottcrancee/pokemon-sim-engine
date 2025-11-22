import os
import re
import time
from collections import deque

from colorama import Fore, Style, init

from envs.pokemon.action import Action
from envs.pokemon.agent import BaseAgent, InputAgent
from envs.pokemon.battle import Battle
from envs.pokemon.pokemon import Pokemon, PokemonStatus
from envs.pokemon.trainer import Trainer

# Initialize colorama
init(autoreset=True)

# Constants for UI
BATTLE_SCREEN_WIDTH = 80
BATTLE_SCREEN_HEIGHT = 24  # Standard terminal height
MESSAGE_BOX_HEIGHT = 7
POKEMON_DISPLAY_HEIGHT = 6
TRAINER_DISPLAY_HEIGHT = 2
STATUS_EFFECTS_HEIGHT = 2
ACTION_MENU_HEIGHT = 5

# Colors
COLOR_HP_HIGH = Fore.GREEN
COLOR_HP_MID = Fore.YELLOW
COLOR_HP_LOW = Fore.RED
COLOR_POKEMON_NAME = Fore.CYAN + Style.BRIGHT
COLOR_TRAINER_NAME = Fore.MAGENTA + Style.BRIGHT
COLOR_MESSAGE = Fore.WHITE
COLOR_WARNING = Fore.YELLOW
COLOR_ERROR = Fore.RED + Style.BRIGHT
COLOR_MENU = Fore.BLUE
COLOR_SELECTED_MENU = Fore.YELLOW + Style.BRIGHT
COLOR_FADED = Fore.BLACK + Style.BRIGHT


def strip_ansi(text: str) -> str:
    """Removes ANSI escape codes from a string."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def right_align(text: str, width: int = BATTLE_SCREEN_WIDTH) -> str:
    """Aligns text to the right, accounting for ANSI codes."""
    visible_length = len(strip_ansi(text))
    padding = max(0, width - visible_length)
    return " " * padding + text


def clear_screen():
    """Clears the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def move_cursor_home():
    """Moves cursor to top-left without clearing screen."""
    print("\033[H", end="")


def draw_box(
    lines: list[str],
    width: int = BATTLE_SCREEN_WIDTH,
    padding: int = 1,
    title: str = None,
    color: str = Fore.WHITE,
) -> list[str]:
    """Draws a box around the given lines of text."""
    box_lines = []
    inner_width = width - 2 - 2 * padding  # Account for borders and padding

    if title:
        # Adjust inner_width for title if it's too long
        if len(title) > inner_width - 2:
            title = title[: inner_width - 5] + "..."
        box_lines.append(
            color
            + "╔"
            + "═" * ((inner_width - len(title)) // 2 - 1)
            + " "
            + title
            + " "
            + "═"
            * (
                (inner_width - len(title)) // 2
                - 1
                + (1 if (inner_width - len(title)) % 2 != 0 else 0)
            )
            + "╗"
            + Style.RESET_ALL
        )
    else:
        box_lines.append(color + "╔" + "═" * (width - 2) + "╗" + Style.RESET_ALL)

    for line in lines:
        # Truncate or pad line to fit within the inner width
        if len(line) > inner_width:
            line = line[:inner_width]
        padded_line = line.ljust(inner_width)
        box_lines.append(
            color + " " * padding + padded_line + " " * padding + Style.RESET_ALL
        )

    box_lines.append(color + "╚" + "═" * (width - 2) + "╝" + Style.RESET_ALL)
    return box_lines


def get_hp_color(current_hp: int, max_hp: int) -> str:
    """Returns a color based on the percentage of HP remaining."""
    percentage = current_hp / max_hp
    if percentage > 0.5:
        return COLOR_HP_HIGH
    elif percentage > 0.2:
        return COLOR_HP_MID
    else:
        return COLOR_HP_LOW


def display_pokemon_stats(pokemon: Pokemon, is_opponent: bool = False) -> list[str]:
    """Generates lines to display a Pokémon's stats."""
    lines = []
    name_line = (
        f"{COLOR_POKEMON_NAME}{pokemon.surname}{Style.RESET_ALL} (L{pokemon.level})"
    )
    hp_bar_length = 20
    current_hp = pokemon.hp
    max_hp = pokemon.max_hp
    hp_color = get_hp_color(current_hp, max_hp)
    hp_percentage = current_hp / max_hp
    filled_length = int(hp_bar_length * hp_percentage)
    hp_bar = (
        hp_color
        + "█" * filled_length
        + Fore.WHITE
        + "░" * (hp_bar_length - filled_length)
        + Style.RESET_ALL
    )
    hp_line = f"HP: {hp_bar} {current_hp}/{max_hp}"

    status_conditions = []
    if pokemon.status != PokemonStatus.Healthy:
        status_conditions.append(str(pokemon.status))
    if pokemon._confused:
        status_conditions.append("Confused")
    if pokemon._taunted:
        status_conditions.append("Taunted")

    status_line = "Status: "
    if status_conditions:
        status_line += ", ".join(status_conditions)
    else:
        status_line += "None"

    if is_opponent:
        lines.append(name_line)
        lines.append(hp_line)
        lines.append(status_line)
    else:
        lines.append(right_align(name_line))
        lines.append(right_align(hp_line))
        lines.append(right_align(status_line))
    return lines


def display_trainer_info(trainer: Trainer, is_opponent: bool = False) -> list[str]:
    """Generates lines to display trainer information."""
    lines = []
    trainer_name = f"{COLOR_TRAINER_NAME}{trainer.name}{Style.RESET_ALL}"

    if is_opponent:
        lines.append(trainer_name)
    else:
        lines.append(right_align(trainer_name))
    return lines


def display_message_box(messages: deque) -> list[str]:
    """Generates lines for the message box."""
    box_content = []
    # Display only the most recent messages that fit
    for msg in list(messages)[-(MESSAGE_BOX_HEIGHT - 2) :]:
        box_content.append(f"{COLOR_MESSAGE}{msg}{Style.RESET_ALL}")
    # Pad with empty lines if there are fewer messages than box height
    while len(box_content) < MESSAGE_BOX_HEIGHT - 2:
        box_content.append("")
    return draw_box(box_content, title="Messages", color=Fore.YELLOW)


def display_action_menu(actions: list[Action], selected_index: int = 0) -> list[str]:
    """Generates lines for the action menu."""
    menu_lines = []
    for i, action in enumerate(actions):
        prefix = f"[{i + 1}] "
        action_str = action.description
        if i == selected_index:
            menu_lines.append(
                f"{COLOR_SELECTED_MENU}> {prefix}{action_str}{Style.RESET_ALL}"
            )
        else:
            menu_lines.append(f"{COLOR_MENU}  {prefix}{action_str}{Style.RESET_ALL}")
    # Pad with empty lines
    while len(menu_lines) < ACTION_MENU_HEIGHT - 2:
        menu_lines.append("")
    return draw_box(menu_lines, title="Actions", color=Fore.BLUE)


def play_ui(battle: Battle, agent_0: BaseAgent, agent_1: BaseAgent):
    message_history = deque(
        maxlen=MESSAGE_BOX_HEIGHT * 2
    )  # Keep more history than displayed

    def add_message(msg: str):
        message_history.append(msg)

    def render_ui(streaming_text: str = None, clear: bool = False):
        if clear:
            clear_screen()
        else:
            move_cursor_home()

        ui_lines = []

        # Opponent's Trainer and Pokemon
        ui_lines.extend(display_trainer_info(battle.trainer_1, is_opponent=True))
        ui_lines.extend(
            display_pokemon_stats(battle.trainer_1.pokemon_team[0], is_opponent=True)
        )
        ui_lines.append("")  # Spacer

        # Player's Trainer and Pokemon
        ui_lines.extend(display_pokemon_stats(battle.trainer_0.pokemon_team[0]))
        ui_lines.extend(display_trainer_info(battle.trainer_0))
        ui_lines.append("")  # Spacer

        # Message Box
        msgs_to_display = list(message_history)
        if streaming_text:
            msgs_to_display.append(streaming_text)

        ui_lines.extend(
            display_message_box(msgs_to_display)
        )  # Display only the most recent messages that fit

        # Pad lines to ensure we overwrite previous content
        final_lines = []
        for line in ui_lines:
            # Calculate visible length to pad correctly
            visible_len = len(strip_ansi(line))
            padding = max(0, BATTLE_SCREEN_WIDTH - visible_len)
            final_lines.append(line + " " * padding)

        # Pad vertically to screen height to clear bottom of screen
        while len(final_lines) < BATTLE_SCREEN_HEIGHT:
            final_lines.append(" " * BATTLE_SCREEN_WIDTH)

        # Print all UI lines
        for line in final_lines:
            print(line)

    def animate_message(text: str):
        # Simple streaming animation
        # Render initial state without clearing (overwriting)
        render_ui(streaming_text="", clear=False)

        for i in range(1, len(text) + 1):
            # Use cursor reset instead of clear for smooth animation
            render_ui(streaming_text=text[:i], clear=False)
            time.sleep(0.01)
        add_message(text)
        # Final render to commit message to history, overwriting
        render_ui(clear=False)

    battle.reset()
    # Clear screen once at the very beginning
    clear_screen()
    animate_message(
        f"Battle started: {battle.trainer_0.name} vs {battle.trainer_1.name}!"
    )
    time.sleep(1)

    while not battle.done:
        # Get actions
        action_0 = None
        action_1 = None

        # Trainer 0's action
        if isinstance(agent_0, InputAgent):
            possible_actions = battle.get_possible_actions(0)
            selected_index = 0
            while action_0 is None:
                clear_screen()
                ui_lines = []
                ui_lines.extend(
                    display_trainer_info(battle.trainer_1, is_opponent=True)
                )
                ui_lines.extend(
                    display_pokemon_stats(
                        battle.trainer_1.pokemon_team[0], is_opponent=True
                    )
                )
                ui_lines.append("")
                ui_lines.extend(display_pokemon_stats(battle.trainer_0.pokemon_team[0]))
                ui_lines.extend(display_trainer_info(battle.trainer_0))
                ui_lines.append("")
                ui_lines.extend(
                    display_message_box(
                        deque(list(message_history)[-MESSAGE_BOX_HEIGHT + 2 :])
                    )
                )
                ui_lines.extend(display_action_menu(possible_actions, selected_index))

                for line in ui_lines:
                    print(line)

                # Handle input for menu navigation
                # This part is tricky for a simple CLI. A more robust solution would use a library like curses.
                # For now, let's simulate a basic input loop.
                print(
                    f"{COLOR_MENU}Use Z/S to navigate, E to select, A to quit.{Style.RESET_ALL}"
                )
                choice = input("Your move: ").lower()

                # Clear screen immediately to remove input line and menu before next render
                clear_screen()

                if choice == "z":
                    selected_index = (selected_index - 1) % len(possible_actions)
                elif choice == "s":
                    selected_index = (selected_index + 1) % len(possible_actions)
                elif choice == "e":
                    action_0 = possible_actions[selected_index]
                elif choice == "a":
                    animate_message(
                        f"{COLOR_ERROR}Player quit the battle.{Style.RESET_ALL}"
                    )
                    battle.winner = 1  # Opponent wins if player quits
                    break
                else:
                    animate_message(
                        f"{COLOR_WARNING}Invalid input. Please use Z, S, E, or A.{Style.RESET_ALL}"
                    )
                    time.sleep(0.5)  # Give user time to read warning
        else:
            action_0 = agent_0.get_action(battle, 0)
            animate_message(f"{battle.trainer_0.name} chose {action_0.description}")
            time.sleep(0.5)  # Small delay for AI action

        if battle.done:  # Check if player quit
            break

        # Trainer 1's action
        if isinstance(agent_1, InputAgent):
            possible_actions = battle.get_possible_actions(1)
            selected_index = 0
            while action_1 is None:
                clear_screen()
                ui_lines = []
                ui_lines.extend(
                    display_trainer_info(battle.trainer_1, is_opponent=True)
                )
                ui_lines.extend(
                    display_pokemon_stats(
                        battle.trainer_1.pokemon_team[0], is_opponent=True
                    )
                )
                ui_lines.append("")
                ui_lines.extend(display_pokemon_stats(battle.trainer_0.pokemon_team[0]))
                ui_lines.extend(display_trainer_info(battle.trainer_0))
                ui_lines.append("")
                ui_lines.extend(
                    display_message_box(
                        deque(list(message_history)[-MESSAGE_BOX_HEIGHT + 2 :])
                    )
                )
                ui_lines.extend(display_action_menu(possible_actions, selected_index))

                for line in ui_lines:
                    print(line)

                print(
                    f"{COLOR_MENU}Use Z/S to navigate, E to select, A to quit.{Style.RESET_ALL}"
                )
                choice = input("Your move: ").lower()

                # Clear screen immediately to remove input line and menu before next render
                clear_screen()

                if choice == "z":
                    selected_index = (selected_index - 1) % len(possible_actions)
                elif choice == "s":
                    selected_index = (selected_index + 1) % len(possible_actions)
                elif choice == "e":
                    action_1 = possible_actions[selected_index]
                elif choice == "a":
                    animate_message(
                        f"{COLOR_ERROR}Player quit the battle.{Style.RESET_ALL}"
                    )
                    battle.winner = 0  # Opponent wins if player quits
                    break
                else:
                    animate_message(
                        f"{COLOR_WARNING}Invalid input. Please use Z, S, E, or A.{Style.RESET_ALL}"
                    )
                    time.sleep(0.5)
        else:
            action_1 = agent_1.get_action(battle, 1)
            animate_message(f"{battle.trainer_1.name} chose {action_1.description}")
            time.sleep(0.5)  # Small delay for AI action

        if battle.done:  # Check if player quit
            break

        # Execute turn
        turn_messages = battle.turn(action_0, action_1)
        for msg in turn_messages:
            animate_message(str(msg))
            time.sleep(1.0)  # Delay for message readability

        # After turn, check for fainted Pokémon and switch if necessary
        # This logic is already handled within battle.turn, but we might want to
        # add specific messages or delays for switches here if battle.turn doesn't
        # provide enough granularity. For now, rely on battle.turn's messages.

        render_ui()  # Final render after all messages for the turn
        time.sleep(1)  # Pause before next turn

    # Battle ended
    clear_screen()

    if battle.tie:
        animate_message(f"{COLOR_WARNING}The battle ended in a tie!{Style.RESET_ALL}")
    elif battle.winner == 0:
        animate_message(
            f"{COLOR_TRAINER_NAME}{battle.trainer_0.name}{Style.RESET_ALL} wins the battle!"
        )
    elif battle.winner == 1:
        animate_message(
            f"{COLOR_TRAINER_NAME}{battle.trainer_1.name}{Style.RESET_ALL} wins the battle!"
        )

    time.sleep(2)  # Final pause
