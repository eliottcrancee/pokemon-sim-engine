import random

from pokemon.action import Action, ActionType
from pokemon.config import DEBUG
from pokemon.item import ItemRegistry
from pokemon.message import Message
from pokemon.move import MoveCategory, MoveRegistry
from pokemon.trainer import Trainer


class Battle:
    __slots__ = (
        "_rng",
        "headless",
        "max_rounds",
        "round",
        "seed",
        "tie",
        "trainers",
        "winner",
    )

    def __init__(
        self,
        trainers: tuple[Trainer, Trainer],
        max_rounds: int = 200,
        headless: bool = True,
        seed: int | None = None,
    ):
        self.trainers = trainers
        self.max_rounds = max_rounds
        self.round = 0
        self.winner = None
        self.tie = False
        self.headless = headless
        self.seed = self._normalize_seed(seed)
        self._rng = random.Random(self.seed)
        self.reset(reseed=False)

    def _normalize_seed(self, seed: int | None) -> int:
        if seed is not None:
            return int(seed)
        # Ensure auto-generated reseeds differ from current seed for deterministic tests.
        candidate = random.getrandbits(64)
        while getattr(self, "seed", None) == candidate:
            candidate = random.getrandbits(64)
        return candidate

    def reseed(self, seed: int | None = None) -> int:
        self.seed = self._normalize_seed(seed)
        self._rng.seed(self.seed)
        return self.seed

    def copy(self):
        cls = self.__class__
        new_battle = cls.__new__(cls)
        new_battle.trainers = (self.trainers[0].copy(), self.trainers[1].copy())
        new_battle.max_rounds = self.max_rounds
        new_battle.round = self.round
        new_battle.winner = self.winner
        new_battle.tie = self.tie
        new_battle.headless = self.headless
        new_battle.seed = self.seed
        new_battle._rng = random.Random()
        new_battle._rng.setstate(self._rng.getstate())
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
            and self.max_rounds == other.max_rounds
            and self.winner == other.winner
            and self.tie == other.tie
        )

    def __hash__(self):
        return hash(
            (
                self.trainers[0],
                self.trainers[1],
                self.round,
                self.max_rounds,
                self.winner,
                self.tie,
            )
        )

    def __str__(self) -> str:
        return f"Battle {self.trainers[0]} VS {self.trainers[1]}"

    def reset(self, reseed: bool = False, seed: int | None = None):
        if seed is not None:
            self.reseed(seed)
        elif reseed:
            self.reseed()

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

        # If the user fainted before this move could be executed
        if not user.is_alive:
            if self.headless:
                return []
            return [
                Message(
                    f"{trainer.name}'s {user.surname} can't move because it fainted!"
                )
            ]

        # If the target fainted before this move could be executed
        if not target.is_alive:
            if self.headless:
                return []
            return [
                Message(f"{user.surname}'s attack missed because the target fainted!")
            ]

        move_slot_index = action.attack_move_slot_index
        if move_slot_index == -1:  # Special case for Struggle
            move = MoveRegistry.get("Struggle")
            if move is None:
                raise ValueError("Struggle move not registered.")
        else:
            move = user.move_slots[move_slot_index].move
            user.decrease_pp(move_slot_index)

        can_attack, new_messages = user.can_attack(rng=self._rng)

        if self.headless:
            messages = []
        else:
            messages = []
            if can_attack:
                messages.append(
                    Message(f"{trainer.name}'s {user.surname} uses {move.name}!")
                )
            messages.extend(new_messages)

        if can_attack:
            damage, base_damage, critical, stab, effectiveness = move.calculate_damage(
                user, target, rng=self._rng
            )
            hp_before = target.hp
            target.hp -= damage
            actual_damage = hp_before - target.hp
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
                    messages.append(
                        Message(f"{target.surname} lost {actual_damage} HP.")
                    )
                elif move.category != MoveCategory.STATUS:
                    messages.append(Message(f"{user.surname}'s attack missed!"))
            messages.extend(
                move.secondary_effect(
                    user,
                    target,
                    damage,
                    base_damage,
                    critical,
                    stab,
                    effectiveness,
                    rng=self._rng,
                )
            )
            if not target.is_alive:
                messages.append(Message(f"{target.surname} fainted!"))
        return messages

    def _execute_switch(
        self, action: Action, trainer: Trainer, opponent: Trainer
    ) -> list[Message]:
        pokemon_index = action.switch_pokemon_index
        target_pokemon = trainer.pokemon_team[pokemon_index]
        if self.headless:
            messages = []
        else:
            messages = [Message(f"{trainer.name} brings in {target_pokemon.surname}!")]
        trainer.switch_pokemon(pokemon_index)
        return messages

    def _execute_use_item(
        self, action: Action, trainer: Trainer, opponent: Trainer
    ) -> list[Message]:
        item_id = action.use_item_id
        target_index = action.use_item_target_index

        item = ItemRegistry.get(item_id)

        target_pokemon = trainer.pokemon_team[target_index]
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
            move_slot_index = action.attack_move_slot_index
            if move_slot_index == -1:  # Struggle has low priority
                return -1
            move = trainer.active_pokemon.move_slots[move_slot_index].move
            return move.priority
        return -10  # PassAction

    def turn(self, action_0: Action, action_1: Action):
        if DEBUG:
            if self.done:
                raise ValueError("Cannot take turn after battle is done")

        self.round += 1
        messages: list[Message] = []

        trainer_0 = self.trainers[0]
        trainer_1 = self.trainers[1]

        p0 = self._get_action_priority(action_0, trainer_0)
        p1 = self._get_action_priority(action_1, trainer_1)

        speed_0 = trainer_0.active_pokemon.speed
        speed_1 = trainer_1.active_pokemon.speed

        if p0 > p1:
            first_action_trainer_id = 0
        elif p1 > p0:
            first_action_trainer_id = 1
        elif speed_0 > speed_1:
            first_action_trainer_id = 0
        elif speed_1 > speed_0:
            first_action_trainer_id = 1
        else:
            first_action_trainer_id = self._rng.choice([0, 1])

        if first_action_trainer_id == 0:
            first_action = action_0
            first_trainer = trainer_0
            first_opponent = trainer_1
            second_action = action_1
            second_trainer = trainer_1
            second_opponent = trainer_0
        else:
            first_action = action_1
            first_trainer = trainer_1
            first_opponent = trainer_0
            second_action = action_0
            second_trainer = trainer_0
            second_opponent = trainer_1

        # Execute first action
        first_messages = self._execute_action(
            first_action, first_trainer, first_opponent
        )
        if not self.headless:
            messages.extend(first_messages)

        if self.end():
            return messages

        # Execute second action
        second_messages = self._execute_action(
            second_action,
            second_trainer,
            second_opponent,
        )
        if not self.headless:
            messages.extend(second_messages)

        if self.end():
            return messages

        # After-turn effects (poison, burn, etc.)
        for pokemon in (trainer_0.active_pokemon, trainer_1.active_pokemon):
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
