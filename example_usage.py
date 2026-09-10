from client import ToTMCTS

def main():
    print("=== Testing Tree of Thoughts MCTS Reasoning ===")
    mcts = ToTMCTS(c_param=1.414)

    # Expand root thoughts
    child = mcts.expand(0, ["Decompose into subproblems", "Direct heuristic guess"])
    print(f"Expanded thoughts from root. First child ID: {child}")

    # Simulate simulation reward
    mcts.backpropagate(child, reward=1.0)

    best_node = mcts.select(0)
    print(f"UCB1 Selected node for next iteration: {best_node}")
    assert best_node is not None
    assert len(mcts.tree) == 3
    print("=== All tests passed successfully! ===")

if __name__ == "__main__":
    main()
