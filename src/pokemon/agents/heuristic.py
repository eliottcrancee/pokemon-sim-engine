import random

from pokemon.action import Action, ActionType
from pokemon.battle import Battle
from pokemon.item import ItemRegistry

from .base_agent import BaseAgent


class RandomAttackAndPotionAgent(BaseAgent):
    """Agent that randomly attacks, but uses a potion if HP is low."""

    def __init__(
        self, name: str = "RandomAttackAndPotion", heal_threshold: float = 0.2
    ):
        name = f"{name}(heal_threshold={heal_threshold})"
        super().__init__(name)
        self.heal_threshold = heal_threshold

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return a random attack or a potion action.
        - If HP is below `heal_threshold`, uses a random available "Potion" item.
        - Otherwise, performs a random attack.
        - If no attack is possible, performs a random action.
        """
        trainer = battle.get_trainer_by_id(trainer_id)
        current_pokemon = trainer.pokemon_team[0]
        actions = battle.get_possible_actions(trainer_id)
        if len(actions) == 1:
            return actions[0]
        if current_pokemon.hp / current_pokemon.max_hp < self.heal_threshold:
            heal_actions = [
                a
                for a in actions
                if a.action_type == ActionType.USE_ITEM
                and "Potion" in ItemRegistry.get(a.item_id).name
            ]
            if heal_actions:
                return random.choice(heal_actions)

        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        if not attack_actions:
            return random.choice(actions)
        return random.choice(attack_actions)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"heal_threshold={self.heal_threshold})"
        )


class BestAttackAgent(BaseAgent):
    """Agent that selects the attack action with the highest power."""

    def __init__(self, name: str = "BestAttack"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return the attack action with the highest base power.
        If no attack is possible, returns a random action.
        """
        actions = battle.get_possible_actions(trainer_id)
        if len(actions) == 1:
            return actions[0]
        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        if not attack_actions:
            return random.choice(actions)

        trainer = battle.get_trainer_by_id(trainer_id)
        pokemon = trainer.pokemon_team[0]

        def get_move_power(action: Action) -> int:
            move = pokemon.move_slots[action.move_slot_index].move
            return move.power or 0

        return max(attack_actions, key=get_move_power)


class BestAttackAndPotionAgent(BaseAgent):
    """Agent that uses the best attack, but uses a potion if HP is low."""

    def __init__(self, name: str = "BestAttackAndPotion", heal_threshold: float = 0.2):
        """Initialize the agent."""
        name = f"{name}(heal_threshold={heal_threshold})"
        super().__init__(name)
        self.heal_threshold = heal_threshold

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return the best attack or a potion action.
        - If HP is below `heal_threshold`, uses the best available "Potion" item.
        - Otherwise, performs the attack with the highest base power.
        - If no attack is possible, performs a random action.
        """
        trainer = battle.get_trainer_by_id(trainer_id)
        current_pokemon = trainer.pokemon_team[0]
        actions = battle.get_possible_actions(trainer_id)
        if len(actions) == 1:
            return actions[0]
        if current_pokemon.hp / current_pokemon.max_hp < self.heal_threshold:
            heal_actions = [
                a
                for a in actions
                if a.action_type == ActionType.USE_ITEM
                and "Potion" in ItemRegistry.get(a.item_id).name
            ]
            if heal_actions:
                return random.choice(heal_actions)

        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        if not attack_actions:
            return random.choice(actions)

        pokemon = trainer.pokemon_team[0]

        def get_move_power(action: Action) -> int:
            move = pokemon.move_slots[action.move_slot_index].move
            return move.power or 0

        return max(attack_actions, key=get_move_power)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"heal_threshold={self.heal_threshold})"
        )


class SmarterHeuristicAgent(BaseAgent):
    """
    A more advanced heuristic agent that scores each action to make a choice.
    - Scores attacks based on power, type effectiveness, and STAB.
    - Scores switching based on type advantages and disadvantages.
    - Scores healing items based on HP restored and avoiding waste.
    """

    def __init__(
        self,
        name: str = "SmarterHeuristic",
        heal_threshold: float = 0.5,
        switch_hp_threshold: float = 0.3,
    ):
        name = f"{name}(heal_t={heal_threshold}, switch_t={switch_hp_threshold})"
        super().__init__(name)
        self.heal_threshold = heal_threshold
        self.switch_hp_threshold = switch_hp_threshold

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        possible_actions = battle.get_possible_actions(trainer_id)
        if len(possible_actions) == 1:
            return possible_actions[0]

        scored_actions = [
            (self._score_action(action, battle, trainer_id), action)
            for action in possible_actions
        ]

        # Choose the action with the highest score.
        # If scores are tied, random.choice will pick one.
        best_score = max(s[0] for s in scored_actions)
        best_actions = [a for s, a in scored_actions if s == best_score]

        return random.choice(best_actions)

    def _score_action(self, action: Action, battle: Battle, trainer_id: int) -> float:
        """Dispatcher to score an action based on its type."""
        if action.action_type == ActionType.ATTACK:
            return self._score_attack(action, battle, trainer_id)
        if action.action_type == ActionType.SWITCH:
            return self._score_switch(action, battle, trainer_id)
        if action.action_type == ActionType.USE_ITEM:
            return self._score_item(action, battle, trainer_id)
        # Pass action
        return 0.0

    def _score_attack(self, action: Action, battle: Battle, trainer_id: int) -> float:
        """Scores an attack based on power, effectiveness, and STAB."""
        user = battle.get_trainer_by_id(trainer_id).active_pokemon
        target = battle.get_trainer_by_id(1 - trainer_id).active_pokemon

        if action.move_slot_index == -1:  # Struggle
            return 1.0  # Low but non-zero score

        move = user.move_slots[action.move_slot_index].move
        if move.power == 0:  # For now, ignore status moves
            return 0.0

        effectiveness = move.type.effectiveness_against(target.types)
        if effectiveness == 0:
            return -1.0  # Heavily penalize moves with no effect

        stab = 1.5 if move.type in user.types else 1.0
        score = move.power * effectiveness * stab

        # Bonus for moves with high priority
        if move.priority > 0:
            score *= 1.2

        return score

    def _score_switch(self, action: Action, battle: Battle, trainer_id: int) -> float:
        """Scores switching to another Pokemon."""
        trainer = battle.get_trainer_by_id(trainer_id)
        opponent = battle.get_trainer_by_id(1 - trainer_id).active_pokemon
        current_pokemon = trainer.active_pokemon
        new_pokemon = trainer.pokemon_team[action.pokemon_index]

        # 1. Calculate opponent's offensive advantage against our current pokemon
        current_disadvantage = max(
            t.effectiveness_against(current_pokemon.types) for t in opponent.types
        )

        # 2. Calculate our new pokemon's defensive advantage against the opponent
        max_effectiveness_against_us = max(
            t.effectiveness_against(new_pokemon.types) for t in opponent.types
        )
        if max_effectiveness_against_us == 0:
            # Full immunity is a huge advantage
            new_defensive_advantage = 10.0
        else:
            new_defensive_advantage = 1 / max_effectiveness_against_us

        # 3. Calculate our new pokemon's offensive advantage against the opponent
        new_offensive_advantage = max(
            t.effectiveness_against(opponent.types) for t in new_pokemon.types
        )

        # Base score is a combination of new advantages minus current disadvantage.
        # Switching has an inherent cost (a free turn for the opponent), so we penalize it slightly.
        score = new_defensive_advantage + new_offensive_advantage - current_disadvantage

        # Strongly incentivize switching if current pokemon is low on health and at a disadvantage
        if (
            current_pokemon.hp / current_pokemon.max_hp < self.switch_hp_threshold
            and current_disadvantage > 1
        ):
            score += 100

        # Penalize switching to a pokemon with a type disadvantage
        if new_defensive_advantage < 1:
            score -= 50

        return score

    def _score_item(self, action: Action, battle: Battle, trainer_id: int) -> float:
        """Scores using a healing item."""
        trainer = battle.get_trainer_by_id(trainer_id)
        item = ItemRegistry.get(action.item_id)
        target = trainer.pokemon_team[action.target_index]

        # Only score healing items for now
        if "Potion" not in item.name and "Heal" not in item.name:
            return 0.0

        if target.hp / target.max_hp > self.heal_threshold:
            return -1.0  # Don't heal if HP is high

        # Simplified scoring: base on item name
        # This is a simple heuristic; a better one would inspect the item's effect.
        if "Hyper Potion" in item.name:
            return 200
        if "Super Potion" in item.name:
            return 150
        if "Potion" in item.name:
            return 100
        if "Full Heal" in item.name and target.status != "HEALTHY":
            return 80  # Curing status is valuable

        return 0.0
