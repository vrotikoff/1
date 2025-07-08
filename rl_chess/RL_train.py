# -*- coding: utf-8 -*-
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
    Создает идеально выровненную ASCII доску с классическими шахматными обозначениями.
    ПРИМЕЧАНИЕ: Для корректного отображения используйте моноширинный шрифт!
    """
    # Маппинг фигур в цветные символы
    piece_symbols = {
        (chess.PAWN, chess.WHITE): 'P', (chess.PAWN, chess.BLACK): 'p',
        (chess.KNIGHT, chess.WHITE): 'N', (chess.KNIGHT, chess.BLACK): 'n',
        (chess.BISHOP, chess.WHITE): 'B', (chess.BISHOP, chess.BLACK): 'b', 
        (chess.ROOK, chess.WHITE): 'R', (chess.ROOK, chess.BLACK): 'r',
        (chess.QUEEN, chess.WHITE): 'Q', (chess.QUEEN, chess.BLACK): 'q',
        (chess.KING, chess.WHITE): 'K', (chess.KING, chess.BLACK): 'k'
    }
    
    lines = []
    lines.append("  +---+---+---+---+---+---+---+---+")
    
    for rank in range(7, -1, -1):  # 8, 7, 6, ..., 1
        rank_line = f"{rank + 1} |"
        for file in range(8):  # a, b, c, ..., h
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                symbol = piece_symbols[(piece.piece_type, piece.color)]
            else:
                symbol = " "
            rank_line += f" {symbol} |"
        lines.append(rank_line)
        lines.append("  +---+---+---+---+---+---+---+---+")
    
    lines.append("    a   b   c   d   e   f   g   h")
    lines.append("")
    lines.append("Обозначения: K=Король Q=Ферзь R=Ладья B=Слон N=Конь P=Пешка")
    lines.append("Белые=ЗАГЛАВНЫЕ, черные=строчные")
    lines.append("")
    lines.append("📋 Для браузера скопируйте и вставьте в <pre> теги или используйте Courier New")
    
    return "\n".join(lines)

def format_board_for_html(board: chess.Board) -> str:
    """
    Создает HTML-версию доски для корректного отображения в браузере.
    """
    # Маппинг фигур в цветные символы
    piece_symbols = {
        (chess.PAWN, chess.WHITE): 'P', (chess.PAWN, chess.BLACK): 'p',
        (chess.KNIGHT, chess.WHITE): 'N', (chess.KNIGHT, chess.BLACK): 'n',
        (chess.BISHOP, chess.WHITE): 'B', (chess.BISHOP, chess.BLACK): 'b', 
        (chess.ROOK, chess.WHITE): 'R', (chess.ROOK, chess.BLACK): 'r',
        (chess.QUEEN, chess.WHITE): 'Q', (chess.QUEEN, chess.BLACK): 'q',
        (chess.KING, chess.WHITE): 'K', (chess.KING, chess.BLACK): 'k'
    }
    
    lines = []
    lines.append('<pre style="font-family: \'Courier New\', Consolas, monospace; font-size: 14px; line-height: 1.2;">')
    lines.append("  +---+---+---+---+---+---+---+---+")
    
    for rank in range(7, -1, -1):  # 8, 7, 6, ..., 1
        rank_line = f"{rank + 1} |"
        for file in range(8):  # a, b, c, ..., h
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                symbol = piece_symbols[(piece.piece_type, piece.color)]
            else:
                symbol = " "
            rank_line += f" {symbol} |"
        lines.append(rank_line)
        lines.append("  +---+---+---+---+---+---+---+---+")
    
    lines.append("    a   b   c   d   e   f   g   h")
    lines.append("")
    lines.append("Обозначения: K=Король Q=Ферзь R=Ладья B=Слон N=Конь P=Пешка")
    lines.append("Белые=ЗАГЛАВНЫЕ, черные=строчные")
    lines.append('</pre>')
    
    return "\n".join(lines)

def create_html_board_file(board: chess.Board, game_num: int, move_num: int, last_move: str):
    """
    Создает HTML файл с текущей доской для просмотра в браузере.
    """
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RL Chess - Игра #{game_num}, Ход #{move_num}</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5; 
        }}
        .header {{ 
            background-color: #2c3e50; 
            color: white; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 20px; 
        }}
        .board {{ 
            background-color: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }}
        pre {{ 
            font-family: 'Courier New', Consolas, monospace; 
            font-size: 16px; 
            line-height: 1.3; 
            margin: 0; 
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 RL Chess Training</h1>
        <h2>Игра #{game_num} | Ход #{move_num} | Последний ход: {last_move}</h2>
    </div>
    <div class="board">
        {format_board_for_html(board)}
    </div>
    <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 8px;">
        <h3>📝 Оптимизации скорости:</h3>
        <ul>
            <li><strong>MCTS симуляции:</strong> 1600 (быстрее в 4x)</li>
            <li><strong>Обновляется каждые {HTML_UPDATE_EVERY_N_MOVES} ходов</strong> во время обучения</li>
            <li><strong>Автоматически обновляется</strong> - просто обновите страницу в браузере</li>
            <li><strong>Моноширинный шрифт</strong> обеспечивает идеальное выравнивание</li>
        </ul>
        <p><em>Файл: current_board.html</em></p>
    </div>
</body>
</html>"""
    
    with open("current_board.html", "w", encoding="utf-8") as f:
        f.write(html_content)

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


# --- Гиперпараметры (оптимизированы для МАКСИМАЛЬНОЙ СКОРОСТИ) ---
NUM_GAMES = 1000  # Количество игр для обучения
LEARNING_RATE = 0.001
BATCH_SIZE = 2048 # Размер батча для обучения нейросети (увеличен для H100: было 600)
MEMORY_SIZE = 15000 # Размер буфера воспроизведения (оптимизировано для скорости: было 20000)
EPOCHS_PER_UPDATE = 2 # Количество эпох обучения на собранных данных (уменьшено для скорости)
GRADIENT_ACCUMULATION_STEPS = 2 # Эмулируем batch_size = 4096 без OOM
SAVE_EVERY_N_GAMES = 25 # Как часто сохранять модель и чекпоинт (реже = быстрее)
AUTO_SAVE_EVERY_N_GAMES = 5 # Автосохранение чекпоинта каждые N игр (без модели)
HTML_UPDATE_EVERY_N_MOVES = 20 # Как часто обновлять HTML файл (было 10)
# Дополнительные оптимизации скорости:
# - MCTS batch_size уменьшен с 64 до 32  
# - Логирование MCTS результатов каждые 400 симуляций
# - Сокращенные логи (топ-3 вместо топ-5 ходов)
# - Доски логируются каждый ход (это не влияет на скорость)
MCTS_SIMULATIONS = 1600 # Количество симуляций MCTS на ход (оптимизировано для скорости: было 6400)
MODEL_SAVE_PATH = "rl_chess_model.pth" # Путь для сохранения модели для игры
CHECKPOINT_PATH = "rl_checkpoint.pth" # Путь для сохранения прогресса обучения

def train():
    """ Главный цикл обучения. """

    # 1. Инициализация
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используется устройство: {device}")
    if device.type == 'cuda':
        logging.info("🚀 АКТИВИРОВАНЫ ОПТИМИЗАЦИИ ДЛЯ МАКСИМАЛЬНОЙ СКОРОСТИ:")
        logging.info("   ⚡ torch.compile() - ожидается 2-3x ускорение")
        logging.info("   🔥 Mixed Precision Training - ускорение ~1.5-2x")
        logging.info("   📊 Gradient Accumulation - эффективный батч 4096")
        logging.info("   🎯 Virtual Loss MCTS - улучшенное исследование дерева")
        logging.info("   ⚡ Оптимизированные настройки скорости:")
        logging.info(f"      🧠 MCTS симуляции: {MCTS_SIMULATIONS} (было 6400)")
        logging.info(f"      💾 MCTS batch_size: 32 (было 64)")
        logging.info(f"      📊 Epochs per update: {EPOCHS_PER_UPDATE} (было 3)")
        logging.info(f"      🗃️ Memory size: {MEMORY_SIZE} (было 20000)")
        logging.info(f"      📝 Доски логируются каждый ход (скорость не страдает)")
        logging.info(f"      🌐 HTML обновления каждые {HTML_UPDATE_EVERY_N_MOVES} ходов")
        logging.info(f"      💾 Полное сохранение каждые {SAVE_EVERY_N_GAMES} игр")
        logging.info(f"      🔄 Автосохранение каждые {AUTO_SAVE_EVERY_N_GAMES} игр")
        logging.info("   🚀 Ожидаемое ускорение: 4-8x (скорость + качество)!")

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

    # 2. УМНАЯ ЗАГРУЗКА ИЗ ЧЕКПОИНТА
    checkpoint_loaded = False
    
    # Проверяем основной чекпоинт
    if os.path.exists(CHECKPOINT_PATH):
        try:
            logging.info(f"🔄 Найден чекпоинт, загрузка прогресса из {CHECKPOINT_PATH}")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            
            # Проверяем целостность чекпоинта
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'game_number', 'replay_memory']
            if all(key in checkpoint for key in required_keys):
                # Загружаем все состояния
                net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_game = checkpoint['game_number']
                replay_memory = checkpoint['replay_memory']
                
                logging.info(f"✅ ЧЕКПОИНТ ЗАГРУЖЕН УСПЕШНО!")
                logging.info(f"   🎮 Игр завершено: {start_game}")
                logging.info(f"   💾 Данных в памяти: {len(replay_memory)}")
                logging.info(f"   🚀 Продолжаем с игры #{start_game + 1}")
                checkpoint_loaded = True
            else:
                logging.warning(f"⚠️ Чекпоинт поврежден (отсутствуют ключи), начинаем заново")
        except Exception as e:
            logging.error(f"❌ Ошибка загрузки чекпоинта: {e}")
            logging.info("🆕 Начинаем новое обучение")
    
    # Проверяем backup чекпоинт если основной не загрузился
    backup_checkpoints = [
        CHECKPOINT_PATH + ".backup",
        CHECKPOINT_PATH + ".auto", 
        CHECKPOINT_PATH + ".direct"
    ]
    
    for backup_path in backup_checkpoints:
        if not checkpoint_loaded and os.path.exists(backup_path):
            try:
                backup_type = backup_path.split('.')[-1]
                logging.info(f"🔄 Пробуем загрузить {backup_type} чекпоинт: {backup_path}")
                checkpoint = torch.load(backup_path, map_location=device)
                
                net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_game = checkpoint['game_number']
                replay_memory = checkpoint['replay_memory']
                
                logging.info(f"✅ {backup_type.upper()} ЧЕКПОИНТ ЗАГРУЖЕН!")
                logging.info(f"   🎮 Игр завершено: {start_game}")
                logging.info(f"   🚀 Продолжаем с игры #{start_game + 1}")
                checkpoint_loaded = True
                break
            except Exception as e:
                logging.error(f"❌ {backup_type} чекпоинт поврежден: {e}")
    
    if not checkpoint_loaded:
        logging.info("🆕 --- НОВОЕ ОБУЧЕНИЕ С НУЛЯ ---")


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
            # Логируем ход и доску (всегда)
            logging.info(f"Игра #{i_game+1} | Ход #{move_counter}: {move.uci()}")
            logging.info(f"\n{format_board_for_log(board)}")
            
            # Создаем HTML файл реже для ускорения
            if move_counter % HTML_UPDATE_EVERY_N_MOVES == 0:
                create_html_board_file(board, i_game+1, move_counter, move.uci())
        
        # Финальная доска игры (всегда)
        logging.info(f"\n{format_board_for_log(board)}")
        create_html_board_file(board, i_game+1, move_counter, "FINAL")
        
        logging.info(f"Игра #{i_game+1} завершена после {move_counter} ходов. Результат: {board.result(claim_draw=True)}")
        
        # Периодическое логирование прогресса
        if (i_game + 1) % 10 == 0:
            progress_pct = (i_game + 1) / NUM_GAMES * 100
            logging.info(f"📊 ПРОГРЕСС: {i_game + 1}/{NUM_GAMES} игр ({progress_pct:.1f}%) | Памяти: {len(replay_memory)}")
        
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
        
        # 5.5. АВТОСОХРАНЕНИЕ ЧЕКПОИНТА (каждые 5 игр для безопасности)
        if (i_game + 1) % AUTO_SAVE_EVERY_N_GAMES == 0 and (i_game + 1) % SAVE_EVERY_N_GAMES != 0:
            try:
                auto_checkpoint_path = CHECKPOINT_PATH + ".auto"
                checkpoint_data = {
                    'game_number': i_game + 1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'replay_memory': replay_memory,
                    'auto_save': True
                }
                torch.save(checkpoint_data, auto_checkpoint_path)
                logging.info(f"🔄 Автосохранение: игра {i_game + 1} → {auto_checkpoint_path}")
            except Exception as e:
                logging.warning(f"⚠️ Автосохранение не удалось: {e}")
        
        # 6. НАДЕЖНОЕ СОХРАНЕНИЕ С BACKUP'АМИ
        if (i_game + 1) % SAVE_EVERY_N_GAMES == 0:
            logging.info(f"💾 --- Сохранение после {i_game + 1} игр ---")
            
            # Сохраняем модель для игры (более надежно)
            try:
                model_temp = MODEL_SAVE_PATH + ".tmp"
                torch.save(net.state_dict(), model_temp)
                # Атомарное переименование (быстро)
                if os.path.exists(MODEL_SAVE_PATH):
                    os.replace(MODEL_SAVE_PATH, MODEL_SAVE_PATH + ".old")
                os.rename(model_temp, MODEL_SAVE_PATH)
                logging.info(f"✅ Модель сохранена: {MODEL_SAVE_PATH}")
            except Exception as e:
                logging.error(f"❌ Ошибка сохранения модели: {e}")
            
            # НАДЕЖНОЕ СОХРАНЕНИЕ ЧЕКПОИНТА
            checkpoint_data = {
                'game_number': i_game + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'replay_memory': replay_memory,
                'save_time': logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None)),
                'total_games': NUM_GAMES,
                'batch_size': BATCH_SIZE,
                'mcts_simulations': MCTS_SIMULATIONS
            }
            
            # Пробуем несколько способов сохранения
            checkpoint_saved = False
            
            # Способ 1: Атомарное сохранение с временным файлом
            try:
                temp_checkpoint = CHECKPOINT_PATH + ".tmp"
                logging.info(f"💾 Сохраняю чекпоинт временно в {temp_checkpoint}")
                torch.save(checkpoint_data, temp_checkpoint)
                
                # Создаем backup предыдущего чекпоинта
                if os.path.exists(CHECKPOINT_PATH):
                    backup_path = CHECKPOINT_PATH + ".backup"
                    os.replace(CHECKPOINT_PATH, backup_path)
                    logging.info(f"📦 Backup создан: {backup_path}")
                
                # Атомарное переименование
                os.rename(temp_checkpoint, CHECKPOINT_PATH)
                logging.info(f"✅ ЧЕКПОИНТ СОХРАНЕН: {CHECKPOINT_PATH}")
                checkpoint_saved = True
                
            except Exception as e:
                logging.error(f"❌ Ошибка атомарного сохранения: {e}")
                
                # Способ 2: Прямое сохранение (если атомарное не сработало)
                try:
                    logging.info("🔄 Пробую прямое сохранение чекпоинта...")
                    torch.save(checkpoint_data, CHECKPOINT_PATH + ".direct")
                    logging.info(f"✅ Прямое сохранение успешно: {CHECKPOINT_PATH}.direct")
                    checkpoint_saved = True
                except Exception as e2:
                    logging.error(f"❌ И прямое сохранение не удалось: {e2}")
            
            if checkpoint_saved:
                memory_mb = len(replay_memory) * 0.001  # Примерная оценка
                logging.info(f"📊 Прогресс: {i_game + 1}/{NUM_GAMES} игр ({(i_game + 1)/NUM_GAMES*100:.1f}%)")
                logging.info(f"💾 Память: {len(replay_memory)} позиций (~{memory_mb:.1f}MB)")
            else:
                logging.error("💥 КРИТИЧНО: Чекпоинт НЕ СОХРАНЕН! Продолжаем обучение...")


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