# -*- coding: utf-8 -*-
import chess

# Размер выхода политики, как в AlphaZero (73 типа ходов * 64 клетки)
POLICY_OUTPUT_SIZE = 4672

# Создаем карты для преобразования ходов в индексы и обратно
MOVE_TO_INDEX_MAP = {}
INDEX_TO_MOVE_MAP = {}

def _build_move_maps():
    """
    Создает детерминированные и быстрые карты для преобразования ходов.
    Логика основана на 73 "плоскостях" для каждого из 64 полей.
    """
    
    # 1. Ходы "королевы" (скольжение по прямым и диагоналям)
    # 8 направлений * 7 возможных расстояний = 56 плоскостей
    # N, NE, E, SE, S, SW, W, NW
    queen_directions = [8, 9, 1, -7, -8, -9, -1, 7] 
    for plane_idx, delta in enumerate(queen_directions):
        for dist in range(1, 8):
            plane = plane_idx * 7 + (dist - 1)
            for from_sq in range(64):
                to_sq = from_sq + delta * dist
                
                # Пропускаем ходы, выходящие за доску
                if not (0 <= to_sq < 64):
                    continue
                # Пропускаем ходы с "перескоком" через край доски
                if max(abs(chess.square_file(from_sq) - chess.square_file(to_sq)),
                       abs(chess.square_rank(from_sq) - chess.square_rank(to_sq))) != dist:
                    continue

                index = plane * 64 + from_sq
                move = chess.Move(from_sq, to_sq)
                MOVE_TO_INDEX_MAP[(from_sq, to_sq, None)] = index
                MOVE_TO_INDEX_MAP[(from_sq, to_sq, chess.QUEEN)] = index
                INDEX_TO_MOVE_MAP[index] = move

    # 2. Ходы коня - 8 плоскостей
    knight_plane_start = 56
    knight_deltas = [17, 15, 10, 6, -6, -10, -15, -17]
    for plane_idx, delta in enumerate(knight_deltas):
        plane = knight_plane_start + plane_idx
        for from_sq in range(64):
            to_sq = from_sq + delta
            if not (0 <= to_sq < 64):
                continue
            if max(abs(chess.square_file(from_sq) - chess.square_file(to_sq)),
                   abs(chess.square_rank(from_sq) - chess.square_rank(to_sq))) != 2:
                continue
                
            index = plane * 64 + from_sq
            move = chess.Move(from_sq, to_sq)
            MOVE_TO_INDEX_MAP[(from_sq, to_sq, None)] = index
            INDEX_TO_MOVE_MAP[index] = move

    # 3. Превращения в "слабые" фигуры - 9 плоскостей
    # (3 фигуры * 3 направления)
    promo_plane_start = 64
    promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    # Направления для белых: NW, N, NE
    promo_deltas_white = [7, 8, 9] 
    for piece_idx, piece in enumerate(promo_pieces):
        for delta_idx, delta in enumerate(promo_deltas_white):
            plane = promo_plane_start + delta_idx * 3 + piece_idx
            # Ходы белых (с 7-й горизонтали)
            for from_file in range(8):
                from_sq = chess.square(from_file, 6)
                if abs(from_file - chess.square_file(from_sq + delta)) <= 1:
                    to_sq = from_sq + delta
                    if 0 <= to_sq < 64 and chess.square_rank(to_sq) == 7:
                        index = plane * 64 + from_sq
                        move = chess.Move(from_sq, to_sq, promotion=piece)
                        MOVE_TO_INDEX_MAP[(from_sq, to_sq, piece)] = index
                        INDEX_TO_MOVE_MAP[index] = move
            # Ходы черных (со 2-й горизонтали)
            for from_file in range(8):
                from_sq = chess.square(from_file, 1)
                # Для черных дельты инвертированы
                if abs(from_file - chess.square_file(from_sq - delta)) <= 1:
                    to_sq = from_sq - delta
                    if 0 <= to_sq < 64 and chess.square_rank(to_sq) == 0:
                        index = plane * 64 + from_sq
                        move = chess.Move(from_sq, to_sq, promotion=piece)
                        MOVE_TO_INDEX_MAP[(from_sq, to_sq, piece)] = index
                        INDEX_TO_MOVE_MAP[index] = move

# Запускаем построение карт при импорте модуля
_build_move_maps()

def move_to_index(move: chess.Move):
    """ Конвертирует объект хода chess.Move в индекс политики. """
    return MOVE_TO_INDEX_MAP.get((move.from_square, move.to_square, move.promotion))

def index_to_move(index, board: chess.Board):
    """ Конвертирует индекс политики в объект хода chess.Move. """
    return INDEX_TO_MOVE_MAP.get(index) 