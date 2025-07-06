import numpy as np
import torch
import chess
import math
import logging

from rl_chess.RL_network import ChessNetwork, board_to_tensor
from rl_chess.RL_utils import move_to_index, index_to_move, POLICY_OUTPUT_SIZE

# Константы для MCTS
C_PUCT = 1.0  # Коэффициент, балансирующий между исследованием (exploration) и использованием (exploitation)

# --- Настройка отдельного логгера для "размышлений" ---
think_logger = logging.getLogger("thinking")
# Чтобы не дублировать хэндлеры при многократном импорте
if not think_logger.handlers:
    fh = logging.FileHandler("thinking_log.txt", mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    think_logger.addHandler(fh)
    think_logger.setLevel(logging.DEBUG)

class Node:
    """ Узел в дереве поиска MCTS. """
    def __init__(self, board: chess.Board, parent=None, prior=1.0):
        self.board = board
        self.parent = parent
        self.children = {}  # Словарь {move: Node}
        self.N = 0  # Количество посещений (visits)
        self.W = 0  # Суммарная ценность (total value)
        self.Q = 0  # Средняя ценность (mean value)
        self.P = prior  # "приоритет" хода, полученный от нейросети

    def expand(self, policy):
        """ Расширение узла: создание дочерних узлов для всех легальных ходов. """
        for move, prob in policy.items():
            if move not in self.children:
                temp_board = self.board.copy()
                temp_board.push(move)
                self.children[move] = Node(board=temp_board, parent=self, prior=prob)
    
    def select_child(self):
        """ Выбор дочернего узла с лучшим UCB1-score (PUCT). """
        best_score = -np.inf
        best_child = None
        best_move = None
        for move, child in self.children.items():
            score = child.get_ucb_score()
            if score > best_score:
                best_score = score
                best_child = child
                best_move = move
        return best_move, best_child

    def update(self, value):
        """ Обновление статистики узла после симуляции. """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

    def get_ucb_score(self):
        """ Расчет PUCT (Polynomial Upper Confidence Trees) - метрики для выбора узла. """
        U = C_PUCT * self.P * (math.sqrt(self.parent.N) / (1 + self.N))
        return self.Q + U
    
    def is_leaf(self):
        return len(self.children) == 0

class MCTSAgent:
    """ Класс для выполнения поиска по дереву Монте-Карло. """
    def __init__(self, network: ChessNetwork, device, num_simulations=1600, batch_size=64):
        self.net = network
        self.device = device
        self.num_simulations = num_simulations
        self.batch_size = batch_size

    def get_move(self, board: chess.Board):
        """
        Выполняет MCTS симуляции, чтобы выбрать лучший ход.
        Возвравращает лучший ход и распределение вероятностей (политику) для обучения.
        """
        root = Node(board.copy(), prior=1.0)
        
        # --- Блок 1: "Интуиция" сети и первая симуляция ---
        # Сразу делаем первую симуляцию для корневого узла.
        # Это даст нам начальную оценку и политику для логирования,
        # а также расширит корень для последующих выборов.
        if board.is_game_over(claim_draw=True):
            sims_done = self.num_simulations # Игра уже закончена
        else:
            values, policies = self.batch_predict([board])
            init_value = values[0]
            init_policy = policies[0]
            
            root.expand(init_policy)
            self.backpropagate(root, init_value)
            sims_done = 1

            # Логирование "интуиции" сети
            top_policy = sorted(init_policy.items(), key=lambda x: x[1], reverse=True)[:5]
            think_logger.debug("[DEBUG] Мнение сети до MCTS:")
            think_logger.debug(f"[DEBUG]   Оценка (Value): {init_value:+.3f}")
            think_logger.debug(f"[DEBUG]   Топ-{len(top_policy)} ходов (Policy):")
            for rank, (mv, prob) in enumerate(top_policy, 1):
                think_logger.debug(f"[DEBUG]     {rank}. {mv.uci()} ({prob:.3f})")
        
        # --- Основной цикл симуляций (пакетная обработка) ---
        while sims_done < self.num_simulations:
            num_to_run = min(self.batch_size, self.num_simulations - sims_done)
            
            leaves_to_process = []
            
            # Фаза 1: Накопление (Selection)
            for _ in range(num_to_run):
                leaf = self.select_leaf(root)
                leaves_to_process.append(leaf)

            boards_to_predict = []
            nodes_for_nn = []
            
            # Обрабатываем терминальные узлы и собираем остальные для нейросети
            for leaf in leaves_to_process:
                if leaf.board.is_game_over(claim_draw=True):
                    # Если узел терминальный, его ценность известна.
                    result = leaf.board.result(claim_draw=True)
                    if result == "1-0": value = 1.0
                    elif result == "0-1": value = -1.0
                    else: value = 0.0
                    # Ценность для игрока, который СДЕЛАЛ ход в это состояние, поэтому инвертируем.
                    self.backpropagate(leaf, -value)
                else:
                    # Если узел не терминальный, добавляем его в очередь на обработку нейросетью.
                    # Проверяем, что он еще не был расширен (это может случиться, если
                    # два потока выбора в одном батче придут в один и тот же лист).
                    if leaf.is_leaf():
                         boards_to_predict.append(leaf.board)
                         nodes_for_nn.append(leaf)
            
            # Фаза 2: Пакетная Оценка (Evaluation)
            if boards_to_predict:
                values, policies = self.batch_predict(boards_to_predict)
                
                # Фаза 3: Расширение и Обратное Распространение (Expansion & Backpropagation)
                for i, node in enumerate(nodes_for_nn):
                    node.expand(policies[i])
                    self.backpropagate(node, values[i])

            sims_done += num_to_run
            
        # Выбираем ход на основе количества посещений
        visit_counts = {move: child.N for move, child in root.children.items()}
        if not visit_counts:
            # Если по какой-то причине нет дочерних узлов, возвращаем случайный ход
            legal_moves = list(board.legal_moves)
            if not legal_moves: # No legal moves, should be caught by is_game_over
                return None, torch.zeros(POLICY_OUTPUT_SIZE, device=self.device)
            return np.random.choice(legal_moves), torch.zeros(POLICY_OUTPUT_SIZE, device=self.device)

        best_move = max(visit_counts, key=visit_counts.get)
        
        # --- Блок 2: Итог размышлений MCTS ---
        top_mcts = sorted(root.children.items(), key=lambda x: x[1].N, reverse=True)[:5]
        think_logger.debug(f"[INFO] Выбран ход: {best_move.uci()}")
        think_logger.debug(f"[DEBUG] Результаты MCTS ({self.num_simulations} симуляций):")
        for rank, (mv, child) in enumerate(top_mcts, 1):
            think_logger.debug(
                f"[DEBUG]   {rank}. Ход: {mv.uci()}, Посещений: {child.N}, Оценка (Q): {child.Q:+.3f}")
        
        # Создаем целевой тензор политики для обучения
        policy_target = torch.zeros(POLICY_OUTPUT_SIZE, device=self.device)
        total_visits = sum(visit_counts.values())
        if total_visits > 0:
            for move, count in visit_counts.items():
                idx = move_to_index(move)
                if idx is not None:
                    policy_target[idx] = count / total_visits
                
        return best_move, policy_target

    def select_leaf(self, node: Node) -> Node:
        """
        Спускается по дереву от корня до листового узла, используя PUCT.
        """
        while not node.is_leaf():
            # Если дочерних узлов нет, но узел не "листовой" (уже был expand),
            # это значит, что ходов из этой позиции нет (пат/мат).
            if not node.children:
                return node
            move, node = node.select_child()
        return node

    def backpropagate(self, node: Node, value: float):
        """
        Распространяет оценку (value) вверх по дереву до корня.
        """
        while node is not None:
            node.update(value)
            value = -value # Инвертируем для родительского узла
            node = node.parent

    def batch_predict(self, boards: list[chess.Board]):
        """
        Получение оценки и политики от нейросети для пакета состояний.
        """
        if not boards:
            return [], []

        self.net.eval()
        with torch.no_grad():
            # Преобразуем все доски в батч тензоров
            state_tensors = [board_to_tensor(b, self.device) for b in boards]
            state_batch = torch.stack(state_tensors)
            
            # Один вызов сети
            policy_logits, values = self.net(state_batch)

        # Обрабатываем результаты
        policies = torch.softmax(policy_logits, dim=1)
        # (N, 1) -> (N) -> list
        values_list = values.squeeze(-1).tolist() 
        
        # Маскируем нелегальные ходы для каждой политики в батче
        final_policies = []
        for i, board in enumerate(boards):
            policy_for_board = policies[i]
            legal_policy = self.mask_illegal_moves(board, policy_for_board)
            final_policies.append(legal_policy)
            
        return values_list, final_policies

    def mask_illegal_moves(self, board: chess.Board, policy):
        """
        Обнуляет вероятности для нелегальных ходов и нормализует остальные.
        """
        policy_dict = {}
        for move in board.legal_moves:
            idx = move_to_index(move)
            if idx is not None:
                policy_dict[move] = policy[idx].item()
        
        sum_probs = sum(policy_dict.values())
        if sum_probs > 0:
            for move in policy_dict:
                policy_dict[move] /= sum_probs
        
        return policy_dict
       