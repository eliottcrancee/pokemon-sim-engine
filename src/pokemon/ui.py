import os
import sys
import re
import time
from collections import deque
import readchar
import readchar.key

from colorama import Fore, Style, init

# Assume these imports exist as per prompt instructions
from pokemon.action import Action, ActionType
from pokemon.agents import BaseAgent, InputAgent
from pokemon.battle import Battle
from pokemon.pokemon import Pokemon, PokemonStatus
from pokemon.trainer import Trainer
from pokemon.item import ItemRegistry # Added for ItemRegistry

# Initialize colorama
init(autoreset=True)

# Constants for UI
BATTLE_SCREEN_WIDTH = 80
BATTLE_SCREEN_HEIGHT = 30
MESSAGE_BOX_HEIGHT = 8
TEXT_DELAY = 0.01  # Seconds between characters for the typewriter effect

# Colors
COLOR_HP_HIGH = Fore.GREEN
COLOR_HP_MID = Fore.YELLOW
COLOR_HP_LOW = Fore.RED
COLOR_HP_EMPTY = Fore.LIGHTBLACK_EX  # Dark Gray for empty HP bar
COLOR_POKEMON_NAME = Fore.CYAN + Style.BRIGHT
COLOR_TRAINER_NAME = Fore.MAGENTA + Style.BRIGHT
COLOR_MESSAGE = Fore.WHITE
COLOR_WARNING = Fore.YELLOW
COLOR_ERROR = Fore.RED + Style.BRIGHT
COLOR_MENU = Fore.BLUE
COLOR_SELECTED_MENU = Fore.YELLOW + Style.BRIGHT
COLOR_FADED = Fore.BLACK + Style.BRIGHT


def get_action_description(action: Action, battle: Battle, trainer_id: int) -> str:
    trainer = battle.get_trainer_by_id(trainer_id)
    if action.action_type == ActionType.ATTACK:
        pokemon = trainer.active_pokemon
        if action.move_slot_index == -1:  # Struggle
            move_name = "Struggle"
            pp_info = "---"
        else:
            move_slot = pokemon.move_slots[action.move_slot_index]
            move_name = move_slot.move.name
            pp_info = f"PP: {move_slot.current_pp}/{move_slot.max_pp}"
        return f"Attack {move_name} ({pp_info})"
    elif action.action_type == ActionType.SWITCH:
        pokemon_to_switch = trainer.pokemon_team[action.pokemon_index]
        return f"Switch to {pokemon_to_switch.surname} (PV: {pokemon_to_switch.hp}/{pokemon_to_switch.max_hp})"
    elif action.action_type == ActionType.USE_ITEM:
        target_pokemon = trainer.pokemon_team[action.target_index]
        item_name = ItemRegistry.get(action.item_id).name
        return f"Use {item_name} on {target_pokemon.surname} (PV: {target_pokemon.hp}/{target_pokemon.max_hp})"
    elif action.action_type == ActionType.PASS:
        return "Wait (Opponent is switching)"
    return str(action)


def strip_ansi(text: str) -> str:
    """Removes ANSI escape codes from a string."""
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def get_visible_length(text: str) -> int:
    """Returns the length of the string excluding ANSI codes."""
    return len(strip_ansi(text))


def pad_visible(text: str, width: int, align_right: bool = False) -> str:
    """Pads a string containing ANSI codes to a specific visual width."""
    vis_len = get_visible_length(text)
    padding_needed = max(0, width - vis_len)

    if align_right:
        return (" " * padding_needed) + text
    else:
        return text + (" " * padding_needed)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def move_cursor_home():
    """Moves cursor to top-left without clearing screen."""
    print("\033[H", end="", flush=True)


def draw_box(
    lines: list[str],
    width: int = BATTLE_SCREEN_WIDTH,
    padding: int = 1,
    title: str = None,
    color: str = Fore.WHITE,
) -> list[str]:
    """Draws a box around the given lines of text."""
    box_lines = []
    inner_width = width - 2 - (2 * padding)

    # --- Top Border ---
    top_border = "╔" + "═" * (width - 2) + "╗"
    if title:
        clean_title = f" {title} "
        title_vis_len = len(clean_title)
        if title_vis_len < (width - 4):
            left_side_len = (width - 2 - title_vis_len) // 2
            right_side_len = (width - 2) - left_side_len - title_vis_len
            top_border = (
                "╔" + ("═" * left_side_len) + clean_title + ("═" * right_side_len) + "╗"
            )

    box_lines.append(color + top_border + Style.RESET_ALL)

    # --- Content ---
    for line in lines:
        if get_visible_length(line) > inner_width:
            line = strip_ansi(line)[:inner_width]

        padded_line = pad_visible(line, inner_width)

        box_lines.append(
            color
            + "║"
            + (" " * padding)
            + padded_line
            + (" " * padding)
            + color
            + "║"
            + Style.RESET_ALL
        )

    # --- Bottom Border ---
    box_lines.append(color + "╚" + "═" * (width - 2) + "╝" + Style.RESET_ALL)
    return box_lines


def get_hp_color(current_hp: int, max_hp: int) -> str:
    if max_hp == 0:
        return COLOR_HP_LOW
    percentage = current_hp / max_hp
    if percentage > 0.5:
        return COLOR_HP_HIGH
    elif percentage > 0.2:
        return COLOR_HP_MID
    else:
        return COLOR_HP_LOW


def display_pokemon_stats(pokemon: Pokemon, is_opponent: bool = False) -> list[str]:
    lines = []
    name_line = (
        f"{COLOR_POKEMON_NAME}{pokemon.surname}{Style.RESET_ALL} (L{pokemon.level})"
    )

    hp_bar_length = 20
    current_hp = max(0, pokemon.hp)
    max_hp = pokemon.max_hp
    hp_color = get_hp_color(current_hp, max_hp)

    if max_hp > 0:
        hp_percentage = current_hp / max_hp
    else:
        hp_percentage = 0

    filled_length = int(hp_bar_length * hp_percentage)

    # --- FIXED: Use solid block with Dark Gray color for the empty part ---
    hp_bar = (
        hp_color
        + "█" * filled_length
        + COLOR_HP_EMPTY  # Dark Gray
        + "█" * (hp_bar_length - filled_length)  # Solid block
        + Style.RESET_ALL
    )

    hp_line = f"HP: {hp_bar} {current_hp}/{max_hp}"

    status_conditions = []
    if pokemon.status != PokemonStatus.HEALTHY:
        status_conditions.append(str(pokemon.status.name))
    if pokemon.confused:
        status_conditions.append("Confused")
    if pokemon.taunted:
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
        lines.append(pad_visible(name_line, BATTLE_SCREEN_WIDTH, align_right=True))
        lines.append(pad_visible(hp_line, BATTLE_SCREEN_WIDTH, align_right=True))
        lines.append(pad_visible(status_line, BATTLE_SCREEN_WIDTH, align_right=True))
    return lines


def display_trainer_info(trainer: Trainer, is_opponent: bool = False) -> list[str]:
    lines = []
    trainer_name = f"{COLOR_TRAINER_NAME}{trainer.name}{Style.RESET_ALL}"
    if is_opponent:
        lines.append(trainer_name)
    else:
        lines.append(pad_visible(trainer_name, BATTLE_SCREEN_WIDTH, align_right=True))
    return lines


def display_message_box(messages: deque) -> list[str]:
    box_content = []
    visible_lines = MESSAGE_BOX_HEIGHT - 2

    msg_list = list(messages)
    display_msgs = (
        msg_list[-visible_lines:] if len(msg_list) > visible_lines else msg_list
    )

    for msg in display_msgs:
        box_content.append(f"{COLOR_MESSAGE}{msg}{Style.RESET_ALL}")

    while len(box_content) < visible_lines:
        box_content.append("")

    return draw_box(box_content, title="Messages", color=Fore.YELLOW)


def display_action_menu(
    actions: list[Action], battle: Battle, trainer_id: int, selected_index: int = 0
) -> list[str]:
    menu_lines = []
    for i, action in enumerate(actions):
        prefix = f"[{i + 1}] "
        action_str = get_action_description(action, battle, trainer_id)

        if i == selected_index:
            menu_lines.append(
                f"{COLOR_SELECTED_MENU}> {prefix}{action_str}{Style.RESET_ALL}"
            )
        else:
            menu_lines.append(f"{COLOR_MENU}  {prefix}{action_str}{Style.RESET_ALL}")

    return draw_box(menu_lines, title="Actions", color=Fore.BLUE)


def play_ui(battle: Battle, agent_0: BaseAgent, agent_1: BaseAgent):
    battle.headless = False
    message_history = deque(maxlen=20)

    def hide_cursor():
        print("\033[?25l", end="", flush=True)

    def show_cursor():
        print("\033[?25h", end="", flush=True)

    def add_message(msg: str):
        message_history.append(msg)

    def render_screen(
        possible_actions=None, selected_index=0, streaming_text=None, input_mode=False
    ):
        """
        Consolidated rendering function.
        """
        move_cursor_home()

        ui_lines = []

        # 1. Opponent Info
        ui_lines.extend(display_trainer_info(battle.trainers[1], is_opponent=True))
        ui_lines.extend(
            display_pokemon_stats(battle.trainers[1].active_pokemon, is_opponent=True)
        )
        ui_lines.append(" " * BATTLE_SCREEN_WIDTH)

        # 2. Player Info
        ui_lines.extend(display_pokemon_stats(battle.trainers[0].active_pokemon))
        ui_lines.extend(display_trainer_info(battle.trainers[0]))
        ui_lines.append(" " * BATTLE_SCREEN_WIDTH)

        # 3. Message Box
        msgs_to_display = deque(list(message_history))
        if streaming_text is not None:
            msgs_to_display.append(streaming_text)
        ui_lines.extend(display_message_box(msgs_to_display))

        # 4. Action Menu (Dynamic)
        if input_mode and possible_actions:
            ui_lines.extend(
                display_action_menu(possible_actions, battle, 0, selected_index)
            )
            ui_lines.append(
                pad_visible(
                    f"{COLOR_MENU}Use arrows to navigate, Enter to select, Q to quit.{Style.RESET_ALL}",
                    BATTLE_SCREEN_WIDTH,
                )
            )

        # 5. Fill remaining screen
        remaining_lines = BATTLE_SCREEN_HEIGHT - len(ui_lines)
        if remaining_lines > 0:
            for _ in range(remaining_lines):
                ui_lines.append(" " * BATTLE_SCREEN_WIDTH)

        # 6. Print
        for line in ui_lines:
            print(pad_visible(line, BATTLE_SCREEN_WIDTH))

        sys.stdout.flush()

    def animate_message(text: str):
        current_text = ""
        for char in text:
            current_text += char
            render_screen(streaming_text=current_text, input_mode=False)
            time.sleep(TEXT_DELAY)

        time.sleep(0.5)
        add_message(text)
        render_screen(input_mode=False)

    hide_cursor()
    try:
        battle.reset()
        clear_screen()
        add_message(
            f"Battle started: {battle.trainers[0].name} vs {battle.trainers[1].name}!"
        )
        render_screen()
        time.sleep(0.5)

        while not battle.done:
            action_0 = None
            action_1 = None

            # --- Trainer 0 (Player) ---
            if isinstance(agent_0, InputAgent):
                possible_actions = battle.get_possible_actions(0)
                selected_index = 0

                while action_0 is None:
                    render_screen(
                        possible_actions=possible_actions,
                        selected_index=selected_index,
                        input_mode=True,
                    )

                    key = readchar.readkey()

                    if key == readchar.key.UP:
                        selected_index = (selected_index - 1) % len(possible_actions)
                    elif key == readchar.key.DOWN:
                        selected_index = (selected_index + 1) % len(possible_actions)
                    elif key == readchar.key.ENTER:
                        action_0 = possible_actions[selected_index]
                    elif key == "q" or key == "\x03":
                        add_message(
                            f"{COLOR_ERROR}Player quit the battle.{Style.RESET_ALL}"
                        )
                        battle.winner = 1
                        return
            else:
                action_0 = agent_0.get_action(battle.copy(), 0, verbose=True)

            if battle.done:
                break

            # --- Trainer 1 (Opponent) ---
            if isinstance(agent_1, InputAgent):
                action_1 = battle.get_possible_actions(1)[0]
            else:
                action_1 = agent_1.get_action(battle.copy(), 1, verbose=True)

            if battle.done:
                break

            # --- Turn Execution ---
            render_screen(input_mode=False)
            turn_messages = battle.turn(action_0, action_1)

            for msg in turn_messages:
                animate_message(str(msg))

            time.sleep(0.5)

        # --- End Game ---
        clear_screen()
        if battle.tie:
            print(f"{COLOR_WARNING}The battle ended in a tie!{Style.RESET_ALL}")
        elif battle.winner == 0:
            print(
                f"{COLOR_TRAINER_NAME}{battle.trainers[0].name}{Style.RESET_ALL} wins the battle!"
            )
        elif battle.winner == 1:
            print(
                f"{COLOR_TRAINER_NAME}{battle.trainers[1].name}{Style.RESET_ALL} wins the battle!"
            )

        time.sleep(2)

    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        print(Style.RESET_ALL)
