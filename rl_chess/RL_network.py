# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

# More neurons for higher strength (requires serious resources)
NUM_RESIDUAL_BLOCKS = 24  # was 20, before 12
NUM_CHANNELS = 384  # was 256 - number of filters in conv layers

class ResidualBlock(nn.Module):
    """
    Residual block, main building unit of the network.
    Consists of two convolutional layers and skip connection.
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
        out += residual # Key element - skip connection
        out = F.relu(out)
        return out

class ChessNetwork(nn.Module):
    """
    Main neural network inspired by AlphaZero architecture.
    """
    def __init__(self):
        super(ChessNetwork, self).__init__()
        # 1. Input layer: converts 18-channel board representation to NUM_CHANNELS
        self.conv_input = nn.Conv2d(18, NUM_CHANNELS, kernel_size=3, stride=1, padding=1)
        self.bn_input = nn.BatchNorm2d(NUM_CHANNELS)

        # 2. Network body: residual blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(NUM_CHANNELS, NUM_CHANNELS) for _ in range(NUM_RESIDUAL_BLOCKS)])

        # 3. Policy Head
        self.policy_conv = nn.Conv2d(NUM_CHANNELS, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        # 4672 - all possible chess moves (including pawn promotions)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4672)

        # 4. Value Head
        self.value_conv = nn.Conv2d(NUM_CHANNELS, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Pass data through input layer
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)

        # Pass through network body
        for block in self.residual_blocks:
            x = block(x)

        # Policy head output
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(-1, 2 * 8 * 8)
        policy = self.policy_fc(policy)

        # Value head output
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(-1, 1 * 8 * 8)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        
        # Return move probabilities (after log_softmax) and position evaluation (after tanh)
        return F.log_softmax(policy, dim=1), torch.tanh(value)

def board_to_tensor(board: chess.Board, device):
    """
    Converts chess.Board to 8x8x18 tensor for neural network input.
    18 channels: 6 for own pieces, 6 for opponent pieces, 6 auxiliary.
    """
    
    # Initialize empty tensor
    tensor = torch.zeros((18, 8, 8), dtype=torch.float32)

    # Dictionary for mapping piece type to channel index
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

    # Fill tensor with piece data
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            # White pieces: channels 0-5, Black: 6-11
            channel_index = piece_to_channel.get(piece_type + (6 if color == chess.BLACK else 0))
            for square in board.pieces(piece_type, color):
                rank, file = chess.square_rank(square), chess.square_file(square)
                tensor[channel_index, rank, file] = 1

    # Channel 12: current player color (0 - black, 1 - white)
    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    # Channel 13: halfmove clock for 50-move rule
    tensor[13, :, :] = board.halfmove_clock / 100.0

    # Castling rights channels
    tensor[14, :, :] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    tensor[15, :, :] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    tensor[16, :, :] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    tensor[17, :, :] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    
    return tensor.to(device) 