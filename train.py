import chess
import numpy as np
import json
from tqdm import tqdm # For progress bars
import multiprocessing # New import
import time # For the delay in the sample game

from ai_core import SimpleAI

WEIGHTS_FILE = "best_weights.json"

# --- Genetic Algorithm Parameters ---
POPULATION_SIZE = 12      # Increased population size for more diversity
GENERATIONS = 15          # More generations for better tuning
MUTATION_RATE = 0.15       # Slightly higher mutation rate
MUTATION_STRENGTH = 0.25   # How much a weight can change during mutation
TOP_N_TO_BREED = 4       # Number of top AIs to select for breeding

# --- Game Parameters ---
# MAX_MOVES_PER_GAME = 100 # To prevent infinitely long games - REMOVED

def print_board(board):
    """Prints the board to the console with coordinates."""
    print("\n  a b c d e f g h")
    print("  -----------------")
    rows = str(board).split('\n')
    for i, row in enumerate(rows):
        print(f"{8 - i} | {row} | {8 - i}")
    print("  -----------------")
    print("  a b c d e f g h\n")

def _play_game_for_pool(args):
    ai1_weights, ai2_weights, verbose = args
    ai1 = SimpleAI(ai1_weights)
    ai2 = SimpleAI(ai2_weights)

    if verbose:
        print("\n--- Sample Game Start ---")
        print("AI 1 (White) weights:", {k: f"{v:.2f}" for k, v in ai1_weights.items()})
        print("AI 2 (Black) weights:", {k: f"{v:.2f}" for k, v in ai2_weights.items()})
    
    board = chess.Board()
    if verbose:
        print_board(board)

    while not board.is_game_over():
        if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
            break

        current_ai = ai1 if board.turn == chess.WHITE else ai2
        move = current_ai.choose_move(board, depth=3)
        
        if move is None:
            break
        
        if verbose:
            print(f"Move {board.fullmove_number}: {move.uci()}")

        board.push(move)

        if verbose:
            print_board(board)

    # --- Определение результата ---
    # Сначала проверяем на пат или мат, так как это окончательные состояния
    if board.is_checkmate():
        # Результат определяется относительно того, чей был ход
        # Если ход белых и мат, то черные выиграли
        result = "0-1" if board.turn == chess.WHITE else "1-0"
    elif board.is_stalemate():
        result = "1/2-1/2"
    # Затем проверяем на ничью по "заявлению"
    elif board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
        result = "1/2-1/2"
    # Также проверяем на другие виды ничьих
    elif board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_seventyfive_moves():
        result = "1/2-1/2"
    else:
        # Если ничего из вышеперечисленного, используем результат по умолчанию.
        # Это может случиться, если цикл прерван move is None
        result = board.result(claim_draw=True)

    if verbose:
        print(f"--- Game Over: {result} ---")

    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    else:
        return 0 # Draw

def create_initial_population():
    """Creates a list of AIs with random weights for material and position."""
    population = []
    for _ in range(POPULATION_SIZE):
        weights = {
            'material': np.random.uniform(0.7, 1.3), # Weight for material advantage
            'position': np.random.uniform(0.0, 0.5),  # Weight for positional advantage (PST)
            'pawn_structure': np.random.uniform(-0.5, 0.0), # Penalize bad pawn structures
            'king_safety': np.random.uniform(0.0, 0.5) # Reward king safety
        }
        population.append(SimpleAI(weights))
    return population

def run_tournament(population):
    """Plays a round-robin tournament and returns fitness scores."""
    fitness_scores = {i: 0 for i in range(len(population))}
    game_tasks = []

    for i in range(len(population)):
        for j in range(len(population)):
            if i == j:
                continue
            # Verbose is now handled outside the tournament loop
            game_tasks.append(((population[i].weights, population[j].weights, False), i, j))

    # Use multiprocessing Pool to run games in parallel
    # The number of processes can be adjusted, or left as default (number of CPU cores)
    with multiprocessing.Pool() as pool:
        with tqdm(total=len(game_tasks), desc="Playing games") as pbar:
            for result_code, i, j in pool.imap_unordered(_play_game_for_pool_wrapper, game_tasks):
                if result_code == 1: # AI1 wins as white
                    fitness_scores[i] += 1
                elif result_code == -1: # AI2 wins as black
                    fitness_scores[j] += 1
                # else: A draw (0) gives no points, so we do nothing.
                pbar.update(1)
                
    return fitness_scores

def _play_game_for_pool_wrapper(args):
    game_args, i, j = args
    result = _play_game_for_pool(game_args)
    return result, i, j

def create_next_generation(population, fitness_scores):
    """Creates a new generation of AIs by breeding the fittest ones."""
    sorted_population = [p for p, f in sorted(zip(population, fitness_scores.values()), key=lambda x: x[1], reverse=True)]
    
    breeders = sorted_population[:TOP_N_TO_BREED]
    next_generation = []

    # Elitism: Keep the very best AI from the current generation
    next_generation.append(breeders[0])

    while len(next_generation) < POPULATION_SIZE:
        parent1, parent2 = np.random.choice(breeders, 2, replace=False)
        
        # Crossover
        child_weights = {k: (parent1.weights[k] + parent2.weights[k]) / 2 for k in parent1.weights}
        
        # Mutation
        if np.random.rand() < MUTATION_RATE:
            for k in child_weights:
                mutation = np.random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
                child_weights[k] += mutation
                if child_weights[k] < 0: child_weights[k] = 0 # Weights shouldn't be negative
        
        next_generation.append(SimpleAI(child_weights))
        
    return next_generation, breeders[0]

if __name__ == "__main__":
    print("Starting enhanced genetic algorithm training...")
    population = create_initial_population()

    # --- Run a single sample game first ---
    print("\nRunning a sample game between two random AIs from generation 0...")
    _play_game_for_pool((population[0].weights, population[1].weights, True))
    print("\nSample game finished. Starting full tournament...")
    # -------------------------------------

    best_ai_overall = None
    best_fitness_overall = -1

    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen + 1}/{GENERATIONS} ---")
        
        fitness_scores = run_tournament(population)
        
        # --- Display Top 3 AIs ---
        sorted_indices = sorted(fitness_scores, key=fitness_scores.get, reverse=True)
        print("\n--- Top 3 AIs of the Generation ---")
        for i in range(min(3, len(sorted_indices))):
            ai_index = sorted_indices[i]
            score = fitness_scores[ai_index]
            weights_str = ", ".join([f"{k}: {v:.3f}" for k, v in population[ai_index].weights.items()])
            print(f"#{i + 1}: Score - {score:.2f}, Weights - {weights_str}")
        print("------------------------------------")
        
        population, best_of_gen = create_next_generation(population, fitness_scores)
        
        current_best_fitness = max(fitness_scores.values())
        print(f"Best fitness in generation: {current_best_fitness:.2f}")
        # Format weights for printing
        weights_str = ", ".join([f"{k}: {v:.3f}" for k, v in best_of_gen.weights.items()])
        print(f"Weights of best AI: {weights_str}")

        if best_ai_overall is None or current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_ai_overall = best_of_gen

    print("\n--- Training Finished ---")
    final_weights_str = ", ".join([f"{k}: {v:.3f}" for k, v in best_ai_overall.weights.items()])
    print(f"Best AI found with fitness: {best_fitness_overall:.2f}")
    print(f"Best weights: {final_weights_str}")

    with open(WEIGHTS_FILE, "w") as f:
        json.dump(best_ai_overall.weights, f)

    print(f"Best AI weights saved to {WEIGHTS_FILE}")