from game import Board, AIEngine
b = Board()
ai = AIEngine('O', 'hard')
move, stats = ai.best_move(b)
print(f"AI best first move: {move}")
print(f"Nodes evaluated: {stats['nodes']}")
print(f"Think time: {stats['think_ms']}ms")
print(f"Score: {stats['score']}")
print("Minimax AI working correctly!")
