import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

# Ещё больше «нейронов» для повышения силы (требует серьёзных ресурсов)
NUM_RESIDUAL_BLOCKS = 24  # было 20, ранее 12
NUM_CHANNELS = 384  # было 256 — количество фильтров в сверточных слоях

class ResidualBlock(nn.Module):
    """
    Остаточный блок, основная строительная единица сети.
    Состоит из двух сверточных слоев и "skip connection".
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual # Ключевой элемент - "проброс" исходных данных
        out = F.relu(out)
        return out

class ChessNetwork(nn.Module):
    """
    Основная нейронная сеть, вдохновленная архитектурой AlphaZero.
    """
    def __init__(self):
        super(ChessNetwork, self).__init__()
        # 1. Входной слой: преобразует 18-канальное представление доски в NUM_CHANNELS каналов
        self.conv_input = nn.Conv2d(18, NUM_CHANNELS, kernel_size=3, stride=1, padding=1)
        self.bn_input = nn.BatchNorm2d(NUM_CHANNELS)

        # 2. "Тело" сети: 6 остаточных блоков
        self.residual_blocks = nn.ModuleList([ResidualBlock(NUM_CHANNELS, NUM_CHANNELS) for _ in range(NUM_RESIDUAL_BLOCKS)])

        # 3. "Голова" Политики (Policy Head)
        self.policy_conv = nn.Conv2d(NUM_CHANNELS, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        # 4672 - это все возможные ходы в шахматах (включая продвижение пешек)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4672)

        # 4. "Голова" Ценности (Value Head)
        self.value_conv = nn.Conv2d(NUM_CHANNELS, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Прогоняем данные через входной слой
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)

        # Прогоняем через тело сети
        for block in self.residual_blocks:
            x = block(x)

        # Выход головы политики
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(-1, 2 * 8 * 8)
        policy = self.policy_fc(policy)

        # Выход головы ценности
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(-1, 1 * 8 * 8)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        
        # Возвращаем вероятности ходов (после log_softmax) и оценку позиции (после tanh)
        return F.log_softmax(policy, dim=1), torch.tanh(value)

def board_to_tensor(board: chess.Board, device):
    """
    Преобразует доску (chess.Board) в тензор 8x8x18 для входа в нейросеть.
    18 каналов: 6 для своих фигур, 6 для фигур оппонента, 6 служебных.
    """
    
    # Инициализируем пустой тензор
    tensor = torch.zeros((18, 8, 8), dtype=torch.float32)

    # Словарь для сопоставления типа фигуры с индексом канала
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
        chess.PAWN + 6: 6,
        chess.KNIGHT + 6: 7,
        chess.BISHOP + 6: 8,
        chess.ROOK + 6: 9,
        chess.QUEEN + 6: 10,
        chess.KING + 6: 11
    }

    # Заполняем тензор данными о фигурах
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            # Белые фигуры: каналы 0-5, Черные: 6-11
            channel_index = piece_to_channel.get(piece_type + (6 if color == chess.BLACK else 0))
            for square in board.pieces(piece_type, color):
                rank, file = chess.square_rank(square), chess.square_file(square)
                tensor[channel_index, rank, file] = 1

    # 12-й канал: цвет текущего игрока (0 - черный, 1 - белый)
    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    # 13-й канал: счетчик ходов без взятий и продвижения пешек (для правила 50 ходов)
    tensor[13, :, :] = board.halfmove_clock / 100.0

    # Каналы для рокировок
    tensor[14, :, :] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    tensor[15, :, :] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    tensor[16, :, :] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    tensor[17, :, :] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    
    return tensor.to(device) 