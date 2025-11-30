import random

from pokemon.action import Action, ActionType
from pokemon.config import DEBUG
from pokemon.item import ItemRegistry
from pokemon.message import Message
from pokemon.move import MoveCategory, MoveRegistry
from pokemon.trainer import Trainer


class Battle:
    __slots__ = ("trainers", "max_rounds", "round", "winner", "tie", "headless")

    def __init__(
        self,
        trainers: tuple[Trainer, Trainer],
        max_rounds: int = 100,
        headless: bool = True,
    ):
        self.trainers = trainers
        self.max_rounds = max_rounds
        self.round = 0
        self.winner = None
        self.tie = False
        self.headless = headless
        self.reset()

    def copy(self):
        cls = self.__class__
        new_battle = cls.__new__(cls)
        new_battle.trainers = (self.trainers[0].copy(), self.trainers[1].copy())
        new_battle.max_rounds = self.max_rounds
        new_battle.round = self.round
        new_battle.winner = self.winner
        new_battle.tie = self.tie
        new_battle.headless = self.headless
        return new_battle

    def get_trainer_by_id(self, trainer_id: int) -> Trainer:
        return self.trainers[trainer_id]

    def __eq__(self, other):
        if not isinstance(other, Battle):
            return NotImplemented
        return (
            self.trainers[0] == other.trainers[0]
            and self.trainers[1] == other.trainers[1]
            and self.round == other.round
        )

    def __hash__(self):
        return hash((self.trainers[0], self.trainers[1], self.round))

    def __str__(self) -> str:
        return f"Battle {self.trainers[0]} VS {self.trainers[1]}"

    def reset(self):
        self.round = 0
        self.winner = None
        self.tie = False
        for trainer in self.trainers:
            trainer.reset()

    def end(self) -> bool:
        if self.trainers[0].is_defeated and self.trainers[1].is_defeated:
            self.tie = True
            return True
        elif self.trainers[0].is_defeated:
            self.winner = 1
            return True
        elif self.trainers[1].is_defeated:
            self.winner = 0
            return True
        elif self.round >= self.max_rounds:
            self.tie = True
            return True
        return False

    @property
    def done(self) -> bool:
        return self.tie or self.winner is not None

    def _execute_action(
        self, action: Action, trainer: Trainer, opponent: Trainer
    ) -> list[Message]:
        action_type = action.action_type
        if action_type == ActionType.PASS:
            return self._execute_pass(action, trainer, opponent)
        if action_type == ActionType.ATTACK:
            return self._execute_attack(action, trainer, opponent)
        if action_type == ActionType.SWITCH:
            return self._execute_switch(action, trainer, opponent)
        if action_type == ActionType.USE_ITEM:
            return self._execute_use_item(action, trainer, opponent)
        raise TypeError(f"Unknown action type: {action.action_type}")

    def _execute_pass(
        self, action: Action, trainer: Trainer, opponent: Trainer
    ) -> list[Message]:
        if self.headless:
            return []
        return [Message(f"{trainer.name} waits.")]

    def _execute_attack(
        self, action: Action, trainer: Trainer, opponent: Trainer
    ) -> list[Message]:
        user = trainer.active_pokemon
        target = opponent.active_pokemon

        # If the target fainted before this move could be executed
        if not target.is_alive:
            if self.headless:
                return []
            return [
                Message(f"{user.surname}'s attack missed because the target fainted!")
            ]

        if action.move_slot_index == -1:  # Special case for Struggle
            move = MoveRegistry.get("Struggle")
            if move is None:
                raise ValueError("Struggle move not registered.")
        else:
            move = user.move_slots[action.move_slot_index].move
            user.decrease_pp(action.move_slot_index)

        if self.headless:
            messages = []
        else:
            messages = [Message(f"{trainer.name}'s {user.surname} uses {move.name}!")]

        can_attack, new_messages = user.can_attack()
        if not self.headless:
            messages.extend(new_messages)

        if can_attack:
            damage, base_damage, critical, stab, effectiveness = move.calculate_damage(
                user, target
            )
            target.hp -= damage
            if not self.headless:
                if effectiveness == 0:
                    messages.append(Message("It doesn't affect the target..."))
                elif damage > 0:
                    if effectiveness > 1:
                        messages.append(Message("It's super effective!"))
                    elif 0 < effectiveness < 1:
                        messages.append(Message("It's not very effective..."))

                    if critical:
                        messages.append(Message("Critical hit!"))
                    messages.append(Message(f"{target.surname} lost {damage} HP."))
                elif move.category != MoveCategory.STATUS:
                    messages.append(Message(f"{user.surname}'s attack missed!"))
            messages.extend(
                move.secondary_effect(
                    user, target, damage, base_damage, critical, stab, effectiveness
                )
            )
            if not target.is_alive:
                messages.append(Message(f"{target.surname} fainted!"))
        return messages

    def _execute_switch(
        self, action: Action, trainer: Trainer, opponent: Trainer
    ) -> list[Message]:
        target_pokemon = trainer.pokemon_team[action.pokemon_index]
        if self.headless:
            messages = []
        else:
            messages = [Message(f"{trainer.name} brings in {target_pokemon.surname}!")]
        trainer.switch_pokemon(action.pokemon_index)
        return messages

    def _execute_use_item(
        self, action: Action, trainer: Trainer, opponent: Trainer
    ) -> list[Message]:
        item = ItemRegistry.get(action.item_id)
        if item is None:
            if self.headless:
                return []
            return [Message(f"Item with if '{action.item_id}' not found.")]

        target_pokemon = trainer.pokemon_team[action.target_index]
        if self.headless:
            messages = []
        else:
            messages = [
                Message(f"{trainer.name} uses {item.name} on {target_pokemon.surname}.")
            ]

        new_messages = item.use(target_pokemon)
        if not self.headless:
            messages.extend(new_messages)
        trainer.decrease_item_quantity(item)

        return messages

    def _get_action_priority(self, action: Action, trainer: Trainer) -> int:
        action_type = action.action_type
        if action_type == ActionType.SWITCH or action_type == ActionType.USE_ITEM:
            return 10  # Highest priority
        if action_type == ActionType.ATTACK:
            if action.move_slot_index == -1:  # Struggle has low priority
                return -1
            move = trainer.active_pokemon.move_slots[action.move_slot_index].move
            return move.priority
        return -10  # PassAction

    def turn(self, action_0: Action, action_1: Action):
        if DEBUG:
            if self.done:
                raise ValueError("Cannot take turn after battle is done")

        self.round += 1
        messages: list[Message] = []

        p0 = self._get_action_priority(action_0, self.trainers[0])
        p1 = self._get_action_priority(action_1, self.trainers[1])

        speed_0 = self.trainers[0].active_pokemon.speed
        speed_1 = self.trainers[1].active_pokemon.speed

        if p0 > p1:
            first_action_trainer_id = 0
        elif p1 > p0:
            first_action_trainer_id = 1
        elif speed_0 > speed_1:
            first_action_trainer_id = 0
        elif speed_1 > speed_0:
            first_action_trainer_id = 1
        else:
            first_action_trainer_id = random.choice([0, 1])

        actions = {0: action_0, 1: action_1}
        trainers = {0: self.trainers[0], 1: self.trainers[1]}
        opponents = {0: self.trainers[1], 1: self.trainers[0]}

        # Execute first action
        first_messages = self._execute_action(
            actions[first_action_trainer_id],
            trainers[first_action_trainer_id],
            opponents[first_action_trainer_id],
        )
        if not self.headless:
            messages.extend(first_messages)

        if self.end():
            return messages

        # Execute second action
        second_action_trainer_id = 1 - first_action_trainer_id
        second_messages = self._execute_action(
            actions[second_action_trainer_id],
            trainers[second_action_trainer_id],
            opponents[second_action_trainer_id],
        )
        if not self.headless:
            messages.extend(second_messages)

        if self.end():
            return messages

        # After-turn effects (poison, burn, etc.)
        for pokemon in [
            self.trainers[0].active_pokemon,
            self.trainers[1].active_pokemon,
        ]:
            if pokemon.is_alive:
                after_turn_messages = pokemon.after_turn()
                if after_turn_messages:
                    if not self.headless:
                        messages.extend(after_turn_messages)
                    if not pokemon.is_alive and not self.headless:
                        messages.append(Message(f"{pokemon.surname} fainted!"))

        self.end()
        return messages

    def get_possible_actions(self, trainer_id: int) -> list[Action]:
        trainer = self.get_trainer_by_id(trainer_id)
        opponent = self.get_trainer_by_id(1 - trainer_id)

        # If this trainer's active pokemon has fainted, they must switch.
        if not trainer.active_pokemon.is_alive:
            actions = [
                Action.create_switch(pokemon_index=i)
                for i, p in enumerate(trainer.pokemon_team)
                if p.is_alive and i > 0
            ]
            # If no pokemon are left to switch to, the battle should end.
            # Returning an empty list signals the trainer is defeated.
            return actions

        # If the opponent's active pokemon has fainted, this trainer must wait.
        if not opponent.active_pokemon.is_alive:
            return [Action.create_pass()]

        # Normal turn actions
        actions: list[Action] = []
        pokemon = trainer.active_pokemon

        can_attack = False
        for move_index, move_slot in enumerate(pokemon.move_slots):
            if move_slot.current_pp > 0:
                if not (
                    move_slot.move.category == MoveCategory.STATUS and pokemon.taunted
                ):
                    actions.append(Action.create_attack(move_slot_index=move_index))
                    can_attack = True

        if not can_attack:  # If no regular attacks available, add Struggle
            actions.append(Action.create_attack(move_slot_index=-1))

        for index, pokemon_in_team in enumerate(trainer.pokemon_team):
            if index > 0 and pokemon_in_team.is_alive:
                actions.append(Action.create_switch(pokemon_index=index))

        for item, quantity in trainer.get_possessed_items():
            for index, pokemon_in_team in enumerate(trainer.pokemon_team):
                if item.can_use(pokemon_in_team):
                    actions.append(
                        Action.create_use_item(item_id=item.id, target_index=index)
                    )

        return actions
