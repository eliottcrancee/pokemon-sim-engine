from pokemon.battle_graph import BattleGraph
from pokemon.battle_registry import BattleRegistry


def main():
    # Use a simple battle for demonstration
    battle_name = "kanto_classic"
    print(f"Initializing battle: {battle_name}")
    battle = BattleRegistry.get(battle_name)

    if not battle:
        print(f"Battle {battle_name} not found.")
        return

    print("Building graph (this may take a while)...")
    # Limit depth and nodes to keep it reasonable for a quick run
    # Reduced samples_per_step to 1 to allow deeper exploration within node limit
    graph = BattleGraph(battle, samples_per_step=1, max_depth=10, max_nodes=1000000)
    graph.build()

    stats = graph.get_stats()
    print("\nGraph Stats:")
    print(f"Total Nodes: {stats['num_nodes']}")
    print(f"Max Depth: {stats['max_depth']}")
    print(f"Max Width: {stats['max_width']}")
    print(f"Depth Distribution: {stats['depth_distribution']}")
    print(f"Stop Reason: {stats['stop_reason']}")


if __name__ == "__main__":
    main()
