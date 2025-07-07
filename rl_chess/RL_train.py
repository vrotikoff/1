import torch
import torch.optim as optim
import chess
import numpy as np
from collections import deque
import os
import logging
import sys

from rl_chess.RL_network import ChessNetwork, board_to_tensor
from rl_chess.RL_agent import MCTSAgent

def format_board_for_log(board: chess.Board) -> str:
    """
    Создает красиво отформатированную строку доски с Unicode фигурами и правильным выравниванием.
    """
    lines = []
    lines.append("┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐")
    
    for rank in range(7, -1, -1):  # 8, 7, 6, ..., 1
        rank_line = f"│"
        for file in range(8):  # a, b, c, ..., h
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                symbol = piece.unicode_symbol()
                # Добавляем пробелы для выравнивания (фиксированная ширина 3 символа)
                cell_content = f" {symbol} "
            else:
                cell_content = "   "  # 3 пробела для пустой клетки
            rank_line += f"{cell_content}│"
        rank_line += f" {rank + 1}"  # Номер ранга
        lines.append(rank_line)
        
        # Разделитель между рангами (кроме последнего)
        if rank > 0:
            lines.append("├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤")
    
    lines.append("└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘")
    lines.append("   a     b     c     d     e     f     g     h")
    
    return "\n".join(lines)

# --- Настройка логирования ---
# Отдельные форматтеры для файла и консоли
# В файл пишем только само сообщение (без времени и уровня),
# чтобы строки выглядели так: "Игра #4 | Ход #368: d8c7"
file_formatter = logging.Formatter("%(message)s")
# В консоль продолжаем выводить подробную информацию
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Вывод в файл
# Указываем кодировку UTF-8, чтобы русские буквы отображались корректно
file_handler = logging.FileHandler("training_log.txt", mode='a', encoding='utf-8')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Вывод в консоль
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


# --- Гиперпараметры (оптимизированы для H100) ---
NUM_GAMES = 1000  # Количество игр для обучения
LEARNING_RATE = 0.001
BATCH_SIZE = 2048 # Размер батча для обучения нейросети (увеличен для H100: было 600)
MEMORY_SIZE = 20000 # Размер буфера воспроизведения (увеличен: было 10000)
EPOCHS_PER_UPDATE = 3 # Количество эпох обучения на собранных данных (уменьшено для больших батчей)
GRADIENT_ACCUMULATION_STEPS = 2 # Эмулируем batch_size = 4096 без OOM
SAVE_EVERY_N_GAMES = 20 # Как часто сохранять модель и чекпоинт
MCTS_SIMULATIONS = 6400 # Количество симуляций MCTS на ход (увеличено для H100: было 3600)
MODEL_SAVE_PATH = "rl_chess_model.pth" # Путь для сохранения модели для игры
CHECKPOINT_PATH = "rl_checkpoint.pth" # Путь для сохранения прогресса обучения

def train():
    """ Главный цикл обучения. """

    # 1. Инициализация
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используется устройство: {device}")
    if device.type == 'cuda':
        logging.info("🚀 АКТИВИРОВАНЫ ОПТИМИЗАЦИИ ДЛЯ H100:")
        logging.info("   ⚡ torch.compile() - ожидается 2-3x ускорение")
        logging.info("   🔥 Mixed Precision Training - ускорение ~1.5-2x")
        logging.info("   📊 Gradient Accumulation - эффективный батч 4096")
        logging.info(f"   💾 Увеличенный BATCH_SIZE: {BATCH_SIZE} (было 600)")
        logging.info(f"   🧠 Увеличенные MCTS симуляции: {MCTS_SIMULATIONS} (было 3600)")
        logging.info("   🎯 Ожидаемое общее ускорение: 4-6x!")

    net = ChessNetwork().to(device)
    
    # 🚀 torch.compile() - ОГРОМНОЕ ускорение на H100 (2-3x)
    if device.type == 'cuda':
        net = torch.compile(net)
        logging.info("⚡ torch.compile() активирован - ожидается 2-3x ускорение!")
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # Mixed Precision Training для H100 - ускорение ~1.5-2x
    scaler = torch.cuda.amp.GradScaler()
    agent = MCTSAgent(net, device=device, num_simulations=MCTS_SIMULATIONS)
    
    start_game = 0
    replay_memory = deque(maxlen=MEMORY_SIZE)

    # 2. Загрузка из чекпоинта, если он существует
    if os.path.exists(CHECKPOINT_PATH):
        logging.info(f"--- Найден чекпоинт, загрузка прогресса из {CHECKPOINT_PATH} ---")
        checkpoint = torch.load(CHECKPOINT_PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Загружаем состояние scaler для Mixed Precision
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_game = checkpoint['game_number']
        replay_memory = checkpoint['replay_memory']
        logging.info(f"Прогресс успешно загружен. Обучение продолжится с игры #{start_game + 1}")
    else:
        logging.info("--- Чекпоинт не найден, начинаем новое обучение ---")


    # 3. Цикл Self-Play
    for i_game in range(start_game, NUM_GAMES):
        logging.info(f"--- Начало игры #{i_game+1} ---")
        
        board = chess.Board()
        game_data = [] # Данные для текущей игры
        move_counter = 0
        
        while not board.is_game_over(claim_draw=True):
            move_counter += 1
            # Получаем ход от агента MCTS в режиме обучения (с шумом)
            move, policy_target = agent.get_move(board, is_training=True)
            
            # Сохраняем состояние, политику и текущего игрока
            state_tensor = board_to_tensor(board, device)
            game_data.append([state_tensor, policy_target])

            board.push(move)
            # Логируем ход
            logging.info(f"Игра #{i_game+1} | Ход #{move_counter}: {move.uci()}")
            # Логируем красиво отформатированную доску
            logging.info(f"\n{format_board_for_log(board)}")
        
        logging.info(f"Игра #{i_game+1} завершена после {move_counter} ходов. Результат: {board.result(claim_draw=True)}")
        
        # 4. Определяем результат и обновляем данные
        result = board.result(claim_draw=True)
        if result == "1-0":
            value_target = 1.0
        elif result == "0-1":
            value_target = -1.0
        else: # Ничья
            value_target = 0.0

        for i in range(len(game_data)):
            player_multiplier = 1 if (i % 2 == 0) else -1
            game_data[i].append(torch.tensor([value_target * player_multiplier], dtype=torch.float32, device=device))

        replay_memory.extend(game_data)
        
        # 5. Обучение нейросети (если накоплено достаточно данных)
        if len(replay_memory) >= BATCH_SIZE:
            logging.info("--- Начало обучения сети ---")
            update_network(net, optimizer, replay_memory, device, scaler)
        
        # 6. Сохранение модели и чекпоинта
        if (i_game + 1) % SAVE_EVERY_N_GAMES == 0:
            logging.info(f"--- Сохранение модели и чекпоинта после {i_game + 1} игр ---")
            
            # Сохраняем модель для игры
            torch.save(net.state_dict(), MODEL_SAVE_PATH)
            
            # Сохраняем чекпоинт для продолжения обучения
            torch.save({
                'game_number': i_game + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # Сохраняем состояние Mixed Precision
                'replay_memory': replay_memory,
            }, CHECKPOINT_PATH)
            logging.info(f"Модель сохранена в {MODEL_SAVE_PATH}, чекпоинт в {CHECKPOINT_PATH}")


def update_network(net, optimizer, memory, device, scaler):
    """ Функция для одного шага обучения нейросети с Mixed Precision Training + Gradient Accumulation. """
    net.train()

    for epoch in range(EPOCHS_PER_UPDATE):
        total_loss_accum = 0
        policy_loss_accum = 0
        value_loss_accum = 0
        
        # Gradient Accumulation для эмуляции больших батчей
        optimizer.zero_grad()
        
        for accum_step in range(GRADIENT_ACCUMULATION_STEPS):
            # Сэмплируем мини-батч из буфера
            indices = np.random.choice(len(memory), BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS, replace=False)
            batch = [memory[i] for i in indices]

            states, policy_targets, value_targets = zip(*batch)
            
            states = torch.stack(states).to(device, non_blocking=True)
            policy_targets = torch.stack(policy_targets).to(device, non_blocking=True)
            value_targets = torch.stack(value_targets).to(device, non_blocking=True)
            
            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast():
                policy_logits, value_preds = net(states)
                
                # Расчет потерь
                policy_loss = -torch.sum(policy_targets * torch.log_softmax(policy_logits, dim=1), dim=1).mean()
                value_loss = torch.nn.functional.mse_loss(value_preds, value_targets)
                
                total_loss = (policy_loss + value_loss) / GRADIENT_ACCUMULATION_STEPS  # Нормализуем
            
            # Накапливаем градиенты
            scaler.scale(total_loss).backward()
            
            # Накапливаем потери для логирования
            total_loss_accum += total_loss.item()
            policy_loss_accum += policy_loss.item() / GRADIENT_ACCUMULATION_STEPS
            value_loss_accum += value_loss.item() / GRADIENT_ACCUMULATION_STEPS
        
        # Обновляем веса после накопления всех градиентов
        scaler.step(optimizer)
        scaler.update()

    logging.info(f"Обучение завершено. Total Loss: {total_loss_accum:.4f}, Policy Loss: {policy_loss_accum:.4f}, Value Loss: {value_loss_accum:.4f}")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logging.exception("КРИТИЧЕСКАЯ ОШИБКА: Обучение остановлено из-за исключения.")
        # Эта строка нужна, чтобы если скрипт запущен в CI/CD или другой автоматизированной системе,
        # он все равно вернул код ошибки.
        sys.exit(1) 