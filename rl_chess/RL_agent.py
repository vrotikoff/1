# -*- coding: utf-8 -*-
import numpy as np
import torch
import chess
import math
import logging

from rl_chess.RL_network import ChessNetwork, board_to_tensor
from rl_chess.RL_utils import move_to_index, index_to_move, POLICY_OUTPUT_SIZE

# Константы для MCTS
C_PUCT = 1.0  # Коэффициент, балансирующий между исследованием (exploration) и использованием (exploitation)
VIRTUAL_LOSS = 1.0  # Величина виртуальной потери для батчевого MCTS (предотвращает повторный выбор одних путей в батче)

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

    def get_move(self, board: chess.Board, is_training=False):
        """
        Выполняет MCTS симуляции, чтобы выбрать лучший ход.
        Возвращает лучший ход и распределение вероятностей (политику) для обучения.
        
        Args:
            board: Текущее состояние шахматной доски
            is_training: Если True, применяется шум к политике для исследования
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
            
            # --- Применение шума к политике в режиме обучения ---
            if is_training:
                # Сохраняем "чистую" политику для логирования
                clean_policy = init_policy.copy()
                
                # Применяем Dirichlet шум
                legal_moves = list(init_policy.keys())
                if len(legal_moves) > 0:
                    # Генерируем шум с параметром alpha = 0.3 (стандартное значение для шахмат)
                    noise = np.random.dirichlet([0.3] * len(legal_moves))
                    
                    # Смешиваем оригинальную политику с шумом (85% оригинал + 15% шум)
                    noisy_policy = {}
                    for i, move in enumerate(legal_moves):
                        noisy_policy[move] = 0.85 * init_policy[move] + 0.15 * noise[i]
                    
                    # Нормализуем, чтобы сумма была равна 1
                    total_prob = sum(noisy_policy.values())
                    if total_prob > 0:
                        for move in noisy_policy:
                            noisy_policy[move] /= total_prob
                    
                    init_policy = noisy_policy
                
                # Логирование сравнения чистой и зашумленной политик
                top_clean = sorted(clean_policy.items(), key=lambda x: x[1], reverse=True)[:5]
                top_noisy = sorted(init_policy.items(), key=lambda x: x[1], reverse=True)[:5]
                
                think_logger.debug("[DEBUG] Применен шум к политике в режиме обучения:")
                think_logger.debug("[DEBUG] Чистая политика (топ-5):")
                for rank, (mv, prob) in enumerate(top_clean, 1):
                    think_logger.debug(f"[DEBUG]   {rank}. {mv.uci()} ({prob:.3f})")
                think_logger.debug("[DEBUG] Зашумленная политика (топ-5):")
                for rank, (mv, prob) in enumerate(top_noisy, 1):
                    think_logger.debug(f"[DEBUG]   {rank}. {mv.uci()} ({prob:.3f})")
            
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
        
        # --- Основной цикл симуляций (пакетная обработка с Virtual Loss) ---
        while sims_done < self.num_simulations:
            num_to_run = min(self.batch_size, self.num_simulations - sims_done)
            
            leaves_to_process = []
            virtual_loss_paths = []  # Сохраняем пути для отмены virtual loss
            
            # Фаза 1: Накопление (Selection) с Virtual Loss
            for _ in range(num_to_run):
                leaf = self.select_leaf(root)
                leaves_to_process.append(leaf)
                
                # Применяем Virtual Loss - временно "наказываем" этот путь
                path_to_leaf = self.get_path_to_node(leaf)
                virtual_loss_paths.append(path_to_leaf)
                self.apply_virtual_loss(path_to_leaf)

            boards_to_predict = []
            nodes_for_nn = []
            terminal_results = []  # Для терминальных узлов
            
            # Обрабатываем терминальные узлы и собираем остальные для нейросети
            for i, leaf in enumerate(leaves_to_process):
                if leaf.board.is_game_over(claim_draw=True):
                    # Если узел терминальный, его ценность известна.
                    result = leaf.board.result(claim_draw=True)
                    if result == "1-0": value = 1.0
                    elif result == "0-1": value = -1.0
                    else: value = 0.0
                    # Ценность для игрока, который СДЕЛАЛ ход в это состояние, поэтому инвертируем.
                    terminal_results.append((i, -value))
                else:
                    # Если узел не терминальный, добавляем его в очередь на обработку нейросетью.
                    # Проверяем, что он еще не был расширен
                    if leaf.is_leaf():
                         boards_to_predict.append(leaf.board)
                         nodes_for_nn.append((i, leaf))  # Сохраняем индекс для связи с virtual loss
            
            # Фаза 2: Пакетная Оценка (Evaluation)
            if boards_to_predict:
                values, policies = self.batch_predict(boards_to_predict)
                
                # Фаза 3: Расширение и Обратное Распространение (Expansion & Backpropagation)
                for j, (original_idx, node) in enumerate(nodes_for_nn):
                    # Сначала отменяем Virtual Loss
                    self.remove_virtual_loss(virtual_loss_paths[original_idx])
                    # Затем расширяем и применяем реальное обновление
                    node.expand(policies[j])
                    self.backpropagate(node, values[j])
            
            # Обрабатываем терминальные узлы
            for original_idx, value in terminal_results:
                # Отменяем Virtual Loss и применяем реальное значение
                self.remove_virtual_loss(virtual_loss_paths[original_idx])
                self.backpropagate(leaves_to_process[original_idx], value)

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

    def get_path_to_node(self, node: Node) -> list[Node]:
        """
        Возвращает путь от корня до указанного узла.
        """
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))  # От корня к листу

    def apply_virtual_loss(self, path: list[Node]):
        """
        Применяет виртуальную потерю к пути от корня до листа.
        Это временно делает путь менее привлекательным для следующих select_leaf.
        """
        for node in path:
            node.N += 1
            node.W -= VIRTUAL_LOSS
            if node.N > 0:
                node.Q = node.W / node.N

    def remove_virtual_loss(self, path: list[Node]):
        """
        Отменяет виртуальную потерю с пути.
        Вызывается перед применением реального обновления.
        """
        for node in path:
            node.N -= 1
            node.W += VIRTUAL_LOSS
            if node.N > 0:
                node.Q = node.W / node.N
            else:
                node.Q = 0

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
        with torch.inference_mode():  # Быстрее чем torch.no_grad()
            # Преобразуем все доски в батч тензоров
            state_tensors = [board_to_tensor(b, self.device) for b in boards]
            state_batch = torch.stack(state_tensors)
            
            # Mixed Precision для inference (дополнительное ускорение)
            with torch.cuda.amp.autocast():
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
       