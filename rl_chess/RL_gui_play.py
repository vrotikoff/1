# -*- coding: utf-8 -*-
import pygame
import chess
import sys
import os
import argparse
import torch
import time

from rl_chess.RL_network import ChessNetwork
from rl_chess.RL_agent import MCTSAgent

# --- Константы ---
SCREEN_SIZE = 600
BOARD_SIZE = 560
MARGIN = (SCREEN_SIZE - BOARD_SIZE) // 2
SQUARE_SIZE = BOARD_SIZE // 8

# --- Настройка Pygame ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("RL Chess AI")

def draw_board(board, selected_square, player_color):
    """ Отрисовывает доску и фигуры. """
    board_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
    font = pygame.font.SysFont(None, 48)
    
    # Инвертируем доску, если игрок играет за черных
    files_range = range(8) if player_color == chess.WHITE else range(7, -1, -1)
    ranks_range = range(7, -1, -1) if player_color == chess.WHITE else range(8)

    for r_idx, r in enumerate(ranks_range):
        for f_idx, f in enumerate(files_range):
            square = chess.square(f, r)
            color = (238, 238, 210) if (r + f) % 2 == 0 else (118, 150, 86)
            
            # Подсветка выбранной клетки и легальных ходов
            if selected_square is not None:
                if square == selected_square:
                    color = (186, 202, 68) # Желтый для выбранной
                else:
                    for move in board.legal_moves:
                        if move.from_square == selected_square and move.to_square == square:
                            # Рисуем кружок для обозначения возможного хода
                            center_pos = (f_idx * SQUARE_SIZE + SQUARE_SIZE // 2, r_idx * SQUARE_SIZE + SQUARE_SIZE // 2)
                            pygame.draw.circle(board_surface, (165, 42, 42, 150), center_pos, SQUARE_SIZE // 4)

            pygame.draw.rect(board_surface, color, (f_idx * SQUARE_SIZE, r_idx * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            piece = board.piece_at(square)
            if piece:
                # Используем стандартные символы Unicode для фигур
                piece_symbol = piece.unicode_symbol()
                # Исправленный цвет: белые фигуры — чёрный, чёрные фигуры — белый
                text_color = (0, 0, 0) if piece.color == chess.WHITE else (255, 255, 255)
                # Подбираем размер шрифта
                piece_font = pygame.font.SysFont("segoeuisymbol", 64)
                text_surface = piece_font.render(piece_symbol, True, text_color)
                text_rect = text_surface.get_rect(center=(f_idx * SQUARE_SIZE + SQUARE_SIZE // 2, r_idx * SQUARE_SIZE + SQUARE_SIZE // 2))
                board_surface.blit(text_surface, text_rect)
                
    return board_surface

def get_square_from_mouse(pos, player_color):
    """ Конвертирует координаты мыши в индекс клетки на доске. """
    x, y = pos
    if not (MARGIN < x < SCREEN_SIZE - MARGIN and MARGIN < y < SCREEN_SIZE - MARGIN):
        return None
    
    file_idx = (x - MARGIN) // SQUARE_SIZE
    rank_idx = (y - MARGIN) // SQUARE_SIZE
    
    if player_color == chess.WHITE:
        return chess.square(file_idx, 7 - rank_idx)
    else:
        return chess.square(7 - file_idx, rank_idx)

def main(model_path, simulations, player_color_choice):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Загрузка ИИ ---
    print("Загрузка RL модели...")
    net = ChessNetwork().to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Исправляем ключи если модель была сохранена с torch.compile() (префиксы "_orig_mod.")
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("Обнаружена модель с torch.compile(), исправляем ключи...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[10:]  # Убираем префикс "_orig_mod."
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        net.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл модели не найден: '{model_path}'")
        sys.exit(1)
    ai = MCTSAgent(net, device=device, num_simulations=simulations)

    # --- Выбор цвета игрока ---
    if player_color_choice is None:
        print("\n🎮 ВЫБОР ЦВЕТА ДЛЯ GUI ИГРЫ:")
        player_color = None
        while player_color not in [chess.WHITE, chess.BLACK]:
            choice = input("Выберите ваш цвет (w - белые, b - черные): ").lower()
            if choice == 'w':
                player_color = chess.WHITE
                print("Вы играете за белых! GUI окно откроется через 2 секунды...")
            elif choice == 'b':
                player_color = chess.BLACK
                print("Вы играете за черных! GUI окно откроется через 2 секунды...")
            else:
                print("Введите 'w' для белых или 'b' для черных")
    else:
        player_color = chess.WHITE if player_color_choice == 'w' else chess.BLACK
        print(f"Цвет задан через командную строку: {'белые' if player_color == chess.WHITE else 'черные'}")
    
    # Небольшая пауза если цвет выбран в консоли
    if player_color_choice is None:
        time.sleep(2)

    # --- Игровое состояние ---
    board = chess.Board()
    selected_square = None
    
    # Обновляем заголовок окна с выбранным цветом
    color_text = "белые" if player_color == chess.WHITE else "черные"
    initial_turn_text = "Ваш ход" if player_color == chess.WHITE else "Ход ИИ"
    pygame.display.set_caption(f"RL Chess AI - Вы играете за {color_text} - {initial_turn_text}")

    # --- Игровой цикл ---
    running = True
    while running:
        is_player_turn = (board.turn == player_color)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and is_player_turn and not board.is_game_over():
                clicked_square = get_square_from_mouse(event.pos, player_color)
                if clicked_square is not None:
                    if selected_square is not None:
                        move = chess.Move(selected_square, clicked_square)
                        # Проверка на превращение пешки
                        if board.piece_at(selected_square).piece_type == chess.PAWN:
                            if (player_color == chess.WHITE and chess.square_rank(clicked_square) == 7) or \
                               (player_color == chess.BLACK and chess.square_rank(clicked_square) == 0):
                                move.promotion = chess.QUEEN 

                        if move in board.legal_moves:
                            board.push(move)
                            selected_square = None
                        else:
                            if board.piece_at(clicked_square) and board.piece_at(clicked_square).color == board.turn:
                                selected_square = clicked_square
                            else:
                                selected_square = None 
                    elif board.piece_at(clicked_square) and board.piece_at(clicked_square).color == board.turn:
                        selected_square = clicked_square

        # --- Ход ИИ ---
        if not is_player_turn and not board.is_game_over():
            pygame.display.set_caption(f"RL Chess AI - Вы играете за {color_text} - ИИ думает...")
            ai_move, _ = ai.get_move(board)
            if ai_move:
                board.push(ai_move)
            pygame.display.set_caption(f"RL Chess AI - Вы играете за {color_text} - Ваш ход")

        # --- Отрисовка ---
        screen.fill((40, 40, 40))
        board_surface = draw_board(board, selected_square, player_color)
        screen.blit(board_surface, (MARGIN, MARGIN))
        
        if board.is_game_over(claim_draw=True):
            font = pygame.font.SysFont(None, 60)
            result = board.result(claim_draw=True)
            outcome = ""
            if board.is_checkmate():
                outcome = "Мат!"
            elif board.is_stalemate():
                outcome = "Пат!"
            elif board.is_insufficient_material():
                outcome = "Ничья (недостаточно фигур)"
            elif board.is_seventyfive_moves():
                outcome = "Ничья (правило 75 ходов)"
            elif board.is_fivefold_repetition():
                outcome = "Ничья (пятикратное повторение)"
            
            text_surface = font.render(f"Игра окончена: {outcome} ({result})", True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(SCREEN_SIZE/2, SCREEN_SIZE/2))
            screen.blit(text_surface, text_rect)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Играть против RL ИИ с графическим интерфейсом.")
    parser.add_argument("--model", type=str, default="rl_chess_model.pth", help="Путь к файлу модели.")
    parser.add_argument("--simulations", type=int, default=200, help="Количество симуляций MCTS за ход.")
    parser.add_argument("--color", type=str, choices=['w', 'b'], default=None, help="Ваш цвет (w - белые, b - черные). Если не указан, будет предложен выбор.")
    args = parser.parse_args()
    
    main(args.model, args.simulations, args.color) 