import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import warnings
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import talib
import ccxt
import time
from datetime import datetime, timedelta
import multiprocessing as mp
import threading
from queue import Queue, Empty
import pickle

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Set the number of threads for PyTorch operations
# Leave some cores for multiprocessing
torch.set_num_threads(max(1, os.cpu_count() // 2))


@dataclass
class TradingConfig:
    """Configuration class for trading parameters"""
    profit_target_min: float = 0.02  # 2% minimum profit target
    profit_target_max: float = 0.04  # 4% maximum profit target
    stop_loss: float = 0.01  # 1% stop loss
    # Maximum position size (as fraction of portfolio)
    max_position_size: float = 1.0
    transaction_cost: float = 0.001  # 0.1% transaction cost
    initial_balance: float = 10000.0  # Initial portfolio balance

    # Multiprocessing parameters
    num_workers: int = 4  # Number of parallel workers
    experience_batch_size: int = 64  # Experiences per worker batch
    queue_maxsize: int = 1000  # Maximum queue size

    # GPU optimization parameters
    gpu_batch_size: int = 512  # Larger batch size for GPU efficiency
    gpu_buffer_size: int = 50000  # Larger buffer for GPU batching
    mixed_precision: bool = True  # Use mixed precision training
    dataloader_workers: int = 2  # DataLoader workers for GPU feeding


class TechnicalIndicators:
    """Class to calculate technical indicators"""

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        return talib.RSI(prices, timeperiod=period)

    @staticmethod
    def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        return talib.SMA(prices, timeperiod=period)

    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        return talib.EMA(prices, timeperiod=period)

    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        return talib.BBANDS(prices, timeperiod=period)

    @staticmethod
    def calculate_volatility(prices: np.ndarray, window: int = 5) -> np.ndarray:
        """Calculate rolling volatility"""
        df = pd.DataFrame(prices)
        return df.rolling(window=window).std().values.flatten()


class FeatureEngineer:
    """Feature engineering class for market data"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.indicators = TechnicalIndicators()
        self.fitted = False

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw market data"""
        df = data.copy()

        # Basic price features
        df['price_change'] = df['Close'].diff()
        df['pct_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['volume_change'] = df['Volume'].pct_change()

        # Technical indicators
        df['rsi'] = self.indicators.calculate_rsi(df['Close'].values)
        df['sma_5'] = self.indicators.calculate_sma(df['Close'].values, 5)
        df['sma_10'] = self.indicators.calculate_sma(df['Close'].values, 10)
        df['sma_20'] = self.indicators.calculate_sma(df['Close'].values, 20)
        df['ema_5'] = self.indicators.calculate_ema(df['Close'].values, 5)
        df['ema_10'] = self.indicators.calculate_ema(df['Close'].values, 10)
        df['ema_20'] = self.indicators.calculate_ema(df['Close'].values, 20)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(
            df['Close'].values)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

        # Volatility
        df['volatility'] = self.indicators.calculate_volatility(
            df['Close'].values)

        # Price position relative to moving averages
        df['price_vs_sma5'] = df['Close'] / df['sma_5'] - 1
        df['price_vs_sma10'] = df['Close'] / df['sma_10'] - 1
        df['price_vs_sma20'] = df['Close'] / df['sma_20'] - 1

        # Volume indicators
        df['volume_sma'] = self.indicators.calculate_sma(
            df['Volume'].values, 10)
        df['volume_ratio'] = df['Volume'] / df['volume_sma']

        # Remove rows with NaN values
        df = df.dropna()

        return df

    def normalize_features(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Normalize features using StandardScaler"""
        if not self.fitted:
            normalized = self.scaler.fit_transform(df[feature_columns])
            self.fitted = True
        else:
            normalized = self.scaler.transform(df[feature_columns])

        return normalized


class ReplayBuffer:
    """Thread-safe experience replay buffer for storing and sampling experiences"""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.max_priority = 1.0
        self.lock = threading.Lock()

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, td_error: float = 1.0):
        """Add experience to buffer with priority"""
        with self.lock:
            experience = (state, action, reward, next_state, done)
            self.buffer.append(experience)
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities.append(priority)
            self.max_priority = max(self.max_priority, priority)

    def add_batch(self, experiences: List[Tuple]):
        """Add batch of experiences to buffer"""
        with self.lock:
            for exp in experiences:
                state, action, reward, next_state, done, td_error = exp
                experience = (state, action, reward, next_state, done)
                self.buffer.append(experience)
                priority = (abs(td_error) + 1e-6) ** self.alpha
                self.priorities.append(priority)
                self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, np.ndarray]:
        """Sample batch with prioritized experience replay"""
        with self.lock:
            if len(self.buffer) < batch_size:
                return [], np.array([])

            # Calculate sampling probabilities
            priorities = np.array(self.priorities)
            probs = priorities / np.sum(priorities)

            # Sample indices
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)

            # Get experiences
            experiences = [self.buffer[i] for i in indices]

            # Calculate importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights /= np.max(weights)

            return experiences, weights

    def __len__(self):
        with self.lock:
            return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network for trading decisions"""

    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [128, 64, 32]):
        super(DQN, self).__init__()

        layers = []
        prev_size = state_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AdvancedDQN(nn.Module):
    """Advanced DQN with Dueling architecture for better GPU utilization"""

    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [512, 512, 256, 128]):
        super(AdvancedDQN, self).__init__()

        # Shared feature extractor
        feature_layers = []
        prev_size = state_size

        for hidden_size in hidden_layers[:-1]:
            feature_layers.append(nn.Linear(prev_size, hidden_size))
            feature_layers.append(nn.LayerNorm(hidden_size))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.Dropout(0.3))
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*feature_layers)

        # Dueling architecture
        final_hidden = hidden_layers[-1]

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, final_hidden),
            nn.LayerNorm(final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, final_hidden),
            nn.LayerNorm(final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, action_size)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for better GPU performance"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class TradingEnvironment:
    """Trading environment for RL agent"""

    def __init__(self, data: pd.DataFrame, config: TradingConfig):
        self.data = data
        self.config = config
        self.current_step = 0
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.balance = config.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0
        self.peak_balance = config.initial_balance

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.balance = self.config.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0
        self.peak_balance = self.config.initial_balance

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step >= len(self.data):
            # +3 for position, balance ratio, steps since entry
            return np.zeros(self.data.shape[1] + 3)

        # Exclude 'Close' price from market state
        market_state = self.data.iloc[self.current_step].drop('Close').values
        position_state = np.array([
            self.position,
            self.balance / self.config.initial_balance,
            # Steps since entry
            (self.current_step - self.current_step) if self.position == 0 else 0
        ])

        return np.concatenate([market_state, position_state])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}

        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close']

        reward = 0
        info = {}

        # Action: 0=hold, 1=buy, 2=sell
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = next_price
            reward = -self.config.transaction_cost  # Transaction cost

        elif action == 2 and self.position == 1:  # Sell
            profit_pct = (next_price - self.entry_price) / self.entry_price
            reward = profit_pct - self.config.transaction_cost

            self.balance *= (1 + profit_pct - self.config.transaction_cost)
            self.total_trades += 1

            if profit_pct >= self.config.profit_target_min:
                reward += 0.1  # Bonus for hitting profit target
                self.winning_trades += 1

            self.position = 0
            self.entry_price = 0

            info = {
                'trade_profit': profit_pct,
                'balance': self.balance,
                'total_trades': self.total_trades,
                'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            }

        # Calculate unrealized P&L if in position
        elif self.position == 1:
            unrealized_pct = (next_price - self.entry_price) / self.entry_price

            # Stop loss check
            if unrealized_pct <= -self.config.stop_loss:
                reward = -self.config.stop_loss - self.config.transaction_cost
                self.balance *= (1 - self.config.stop_loss -
                                 self.config.transaction_cost)
                self.position = 0
                self.entry_price = 0
                self.total_trades += 1

            # Profit target check
            elif unrealized_pct >= self.config.profit_target_max:
                reward = self.config.profit_target_max - \
                    self.config.transaction_cost + 0.2  # Extra bonus
                self.balance *= (1 + self.config.profit_target_max -
                                 self.config.transaction_cost)
                self.position = 0
                self.entry_price = 0
                self.total_trades += 1
                self.winning_trades += 1

            else:
                reward = unrealized_pct * 0.1  # Small reward for unrealized gains

        # Update max drawdown
        self.peak_balance = max(self.peak_balance, self.balance)
        current_drawdown = (self.peak_balance -
                            self.balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self._get_state(), reward, done, info


def run_worker(worker_id: int, data_chunk: pd.DataFrame, config: TradingConfig,
               experience_queue: mp.Queue, episodes_per_worker: int, state_size: int):
    """Worker function to generate experiences in parallel"""
    try:
        # Create local environment and simple policy for experience generation
        env = TradingEnvironment(data_chunk, config)

        print(
            f"Worker {worker_id} starting with {len(data_chunk)} data points")

        for episode in range(episodes_per_worker):
            state = env.reset()
            episode_experiences = []

            while True:
                # Simple epsilon-greedy action selection for experience generation
                if random.random() < 0.3:  # 30% exploration
                    action = random.randint(0, 2)
                else:
                    # Simple heuristic policy based on basic indicators
                    if len(env.data) > env.current_step:
                        close_price = env.data.iloc[env.current_step]['Close']
                        if env.current_step > 0:
                            prev_close = env.data.iloc[env.current_step - 1]['Close']
                            if close_price > prev_close * 1.01:  # Price up 1%
                                action = 1  # Buy
                            elif close_price < prev_close * 0.99:  # Price down 1%
                                action = 2  # Sell
                            else:
                                action = 0  # Hold
                        else:
                            action = 0
                    else:
                        action = 0

                next_state, reward, done, info = env.step(action)

                # Calculate simple TD error estimate
                td_error = abs(reward) + 0.1

                experience = (state.copy(), action, reward,
                              next_state.copy(), done, td_error)
                episode_experiences.append(experience)

                state = next_state

                if done:
                    break

            # Send experiences to queue in batches
            if episode_experiences:
                try:
                    experience_queue.put(
                        (worker_id, episode_experiences), timeout=1.0)
                except:
                    pass  # Queue might be full, skip this batch

            if episode % 10 == 0:
                print(
                    f"Worker {worker_id} completed episode {episode}/{episodes_per_worker}")

        print(f"Worker {worker_id} completed all episodes")

    except Exception as e:
        print(f"Worker {worker_id} error: {e}")


class MultiprocessingCryptoRLTrader:
    """Main RL trader class with multiprocessing and GPU optimization capabilities"""

    def __init__(self, state_size: int, action_size: int = 3, config: TradingConfig = None, use_advanced_dqn: bool = True):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or TradingConfig()
        self.use_advanced_dqn = use_advanced_dqn

        # GPU optimization
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Optimize GPU memory allocation
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

        # Neural networks - larger for better GPU utilization
        if use_advanced_dqn:
            self.q_network = AdvancedDQN(
                state_size, action_size).to(self.device)
            self.target_network = AdvancedDQN(
                state_size, action_size).to(self.device)
        else:
            self.q_network = DQN(state_size, action_size).to(self.device)
            self.target_network = DQN(state_size, action_size).to(self.device)

        # Mixed precision training for better GPU utilization
        self.use_mixed_precision = self.config.mixed_precision and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None

        # Optimizers with better GPU settings
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=0.001,
            weight_decay=1e-4,
            eps=1e-4  # Better for mixed precision
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=100
        )

        # Larger replay buffer for GPU batching
        self.memory = ReplayBuffer(capacity=self.config.gpu_buffer_size)

        # Training parameters optimized for GPU
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = self.config.gpu_batch_size  # Much larger batch size
        self.update_target_freq = 500  # More frequent updates with larger batches
        self.learn_freq = 2  # More frequent learning
        self.step_count = 0

        # Performance tracking
        self.training_rewards = []
        self.training_losses = []
        self.performance_history = []
        self.gpu_utilization = []

        # Multiprocessing
        self.experience_queue = mp.Queue(maxsize=self.config.queue_maxsize)
        self.workers = []
        self.experience_collector_thread = None
        self.training_active = False

        # GPU batch processing
        self.experience_buffer = []
        self.buffer_lock = threading.Lock()

        # Update target network
        self.update_target_network()

        # Print model info
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(
            p.numel() for p in self.q_network.parameters() if p.requires_grad)
        print(
            f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"Batch size: {self.batch_size}")
        print(f"Mixed precision: {self.use_mixed_precision}")

    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Get action using epsilon-greedy policy with GPU optimization"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(
                0).to(self.device, non_blocking=True)

            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    q_values = self.q_network(state_tensor)
            else:
                q_values = self.q_network(state_tensor)

        return q_values.argmax().item()

    def get_action_batch(self, states: np.ndarray) -> np.ndarray:
        """Batch action prediction for better GPU utilization"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(
                self.device, non_blocking=True)

            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    q_values = self.q_network(states_tensor)
            else:
                q_values = self.q_network(states_tensor)

        return q_values.argmax(dim=1).cpu().numpy()

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        # Calculate TD error for prioritized replay
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(
                0).to(self.device, non_blocking=True)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(
                0).to(self.device, non_blocking=True)

            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    current_q = self.q_network(state_tensor)[0][action]
                    if not done:
                        next_q = self.target_network(
                            next_state_tensor).max(1)[0]
                        target_q = reward + 0.99 * next_q
                    else:
                        target_q = reward
            else:
                current_q = self.q_network(state_tensor)[0][action]
                if not done:
                    next_q = self.target_network(next_state_tensor).max(1)[0]
                    target_q = reward + 0.99 * next_q
                else:
                    target_q = reward

            td_error = abs(current_q - target_q).item()

        self.memory.add(state, action, reward, next_state, done, td_error)

    def experience_collector(self):
        """Collect experiences from worker processes with GPU optimization"""
        print("GPU-optimized experience collector thread started")

        while self.training_active:
            try:
                # Get experience batch from queue
                worker_id, experiences = self.experience_queue.get(timeout=1.0)

                with self.buffer_lock:
                    self.experience_buffer.extend(experiences)

                    # Process in larger batches for GPU efficiency
                    if len(self.experience_buffer) >= self.batch_size:
                        batch_to_add = self.experience_buffer[:self.batch_size]
                        self.experience_buffer = self.experience_buffer[self.batch_size:]

                        # Add batch to replay buffer
                        self.memory.add_batch(batch_to_add)

                        print(f"GPU batch processed: {len(batch_to_add)} experiences, "
                              f"Buffer size: {len(self.memory)}")

            except:
                continue  # Timeout or queue empty

        print("Experience collector thread stopped")

    def replay(self) -> float:
        """GPU-optimized training on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0

        # Sample larger batch for GPU efficiency
        experiences, weights = self.memory.sample(self.batch_size)

        if not experiences:
            return 0

        # Prepare tensors for GPU with non_blocking transfer
        states = torch.FloatTensor([e[0] for e in experiences]).to(
            self.device, non_blocking=True)
        actions = torch.LongTensor([e[1] for e in experiences]).to(
            self.device, non_blocking=True)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(
            self.device, non_blocking=True)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(
            self.device, non_blocking=True)
        dones = torch.BoolTensor([e[4] for e in experiences]).to(
            self.device, non_blocking=True)
        weights_tensor = torch.FloatTensor(
            weights).to(self.device, non_blocking=True)

        # Forward pass with mixed precision
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                # Current Q values
                current_q_values = self.q_network(
                    states).gather(1, actions.unsqueeze(1))

                # Next Q values from target network
                with torch.no_grad():
                    next_q_values = self.target_network(next_states).max(1)[0]
                    target_q_values = rewards + (0.99 * next_q_values * ~dones)

                # Calculate loss with importance sampling weights
                loss = F.mse_loss(current_q_values.squeeze(),
                                  target_q_values, reduction='none')
                weighted_loss = (loss * weights_tensor).mean()

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(weighted_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular precision training
            current_q_values = self.q_network(
                states).gather(1, actions.unsqueeze(1))

            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (0.99 * next_q_values * ~dones)

            loss = F.mse_loss(current_q_values.squeeze(),
                              target_q_values, reduction='none')
            weighted_loss = (loss * weights_tensor).mean()

            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Update learning rate based on loss
        self.scheduler.step(weighted_loss.item())

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return weighted_loss.item()

    def start_workers(self, train_data: pd.DataFrame, episodes: int):
        """Start worker processes for experience generation"""
        print(f"Starting {self.config.num_workers} workers...")

        # Split data among workers
        data_splits = np.array_split(train_data, self.config.num_workers)
        episodes_per_worker = episodes // self.config.num_workers

        self.training_active = True

        # Start experience collector thread
        self.experience_collector_thread = threading.Thread(
            target=self.experience_collector, daemon=True)
        self.experience_collector_thread.start()

        # Start worker processes
        for i in range(self.config.num_workers):
            worker = mp.Process(
                target=run_worker,
                args=(i, data_splits[i], self.config, self.experience_queue,
                      episodes_per_worker, self.state_size),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        print(f"Started {len(self.workers)} worker processes")

    def stop_workers(self):
        """Stop all worker processes"""
        print("Stopping workers...")

        self.training_active = False

        # Terminate worker processes
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)

        # Join experience collector thread
        if self.experience_collector_thread and self.experience_collector_thread.is_alive():
            self.experience_collector_thread.join(timeout=5)

        self.workers.clear()
        print("All workers stopped")

    def train(self, train_data: pd.DataFrame, episodes: int, save_path: str = "crypto_rl_model.pth"):
        """Train the RL agent with multiprocessing"""
        print(f"Starting multiprocessing training with {episodes} episodes...")

        best_reward = float('-inf')
        training_start_time = time.time()

        try:
            # Start worker processes
            self.start_workers(train_data, episodes)

            # Training loop
            print("Waiting for initial experiences...")

            # Wait for some initial experiences
            while len(self.memory) < self.batch_size * 4:
                time.sleep(1)
                print(f"Buffer size: {len(self.memory)}")

            print("Starting model training...")

            training_steps = 0
            last_save_time = time.time()

            # Train while workers are generating experiences
            while any(worker.is_alive() for worker in self.workers) or not self.experience_queue.empty():

                # Train the model
                if len(self.memory) >= self.batch_size:
                    loss = self.replay()
                    self.training_losses.append(loss)
                    training_steps += 1

                    # Update target network
                    if training_steps % self.update_target_freq == 0:
                        self.update_target_network()
                        print(
                            f"Target network updated at step {training_steps}")

                    # Periodic progress update
                    if training_steps % 100 == 0:
                        avg_loss = np.mean(
                            self.training_losses[-100:]) if self.training_losses else 0
                        elapsed_time = time.time() - training_start_time
                        print(f"Training step {training_steps}, Avg Loss: {avg_loss:.4f}, "
                              f"Buffer size: {len(self.memory)}, Epsilon: {self.epsilon:.4f}, "
                              f"Elapsed: {elapsed_time:.1f}s")

                # Save model periodically
                if time.time() - last_save_time > 300:  # Every 5 minutes
                    self.save_model(save_path.replace(
                        '.pth', '_checkpoint.pth'))
                    last_save_time = time.time()

                time.sleep(0.01)  # Small delay to prevent CPU spinning

            print("All workers completed. Finishing training...")

            # Continue training on remaining experiences
            final_training_steps = 0
            while len(self.memory) >= self.batch_size and final_training_steps < 1000:
                loss = self.replay()
                self.training_losses.append(loss)
                final_training_steps += 1

                if final_training_steps % 100 == 0:
                    avg_loss = np.mean(self.training_losses[-100:])
                    print(
                        f"Final training step {final_training_steps}, Loss: {avg_loss:.4f}")

            # Save final model
            self.save_model(save_path)

            total_time = time.time() - training_start_time
            print(f"Training completed in {total_time:.1f}s")
            print(
                f"Total training steps: {training_steps + final_training_steps}")
            print(f"Final buffer size: {len(self.memory)}")

        except Exception as e:
            print(f"Training error: {e}")
            raise
        finally:
            self.stop_workers()

    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'training_losses': self.training_losses
        }, path)

    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(
            checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.step_count = checkpoint.get('step_count', 0)
        self.training_losses = checkpoint.get('training_losses', [])

    def evaluate(self, env: TradingEnvironment) -> Dict:
        """Evaluate the trained model"""
        state = env.reset()
        total_reward = 0
        trades = []

        while True:
            action = self.get_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            total_reward += reward

            if info:
                trades.append(info)

            state = next_state

            if done:
                break

        return {
            'total_reward': total_reward,
            'final_balance': env.balance,
            'total_trades': env.total_trades,
            'winning_trades': env.winning_trades,
            'win_rate': env.winning_trades / env.total_trades if env.total_trades > 0 else 0,
            'max_drawdown': env.max_drawdown,
            'return_pct': (env.balance - env.config.initial_balance) / env.config.initial_balance,
            'trades': trades
        }

    def continuous_learning_update(self, new_data: pd.DataFrame, learning_rate_factor: float = 0.1):
        """Update model with new data for continuous learning"""
        # Reduce learning rate for stability
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= learning_rate_factor

        # Create temporary environment for new data
        temp_env = TradingEnvironment(new_data, self.config)

        # Train on new data for a few episodes
        for _ in range(10):
            state = temp_env.reset()
            while True:
                action = self.get_action(state, training=True)
                next_state, reward, done, info = temp_env.step(action)

                self.remember(state, action, reward, next_state, done)

                # Learn from new experience
                if len(self.memory) >= self.batch_size:
                    self.replay()

                state = next_state
                if done:
                    break

        # Restore original learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= learning_rate_factor


class RealTimeTrader:
    """Real-time trading implementation with live data using CCXT"""

    def __init__(self, model_path: str, symbol: str = "BTC/USDT", exchange: str = "coinbase", config: TradingConfig = None):
        self.symbol = symbol
        self.exchange_name = exchange
        self.config = config or TradingConfig()
        self.model_path = model_path

        # Initialize CCXT exchange
        self.exchange = self._initialize_exchange()

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()

        # Load trained model
        self.trader = None
        self.feature_columns = None

        # Trading state
        self.position = 0
        self.entry_price = 0
        self.balance = self.config.initial_balance
        self.trade_history = []

        # Data buffer for feature calculation
        self.data_buffer = deque(maxlen=100)  # Keep last 100 minutes of data

    def _initialize_exchange(self):
        """Initialize CCXT exchange"""
        try:
            # Initialize exchange (using public data only)
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange = exchange_class({
                'apiKey': '',
                'secret': '',
                'timeout': 30000,
                'enableRateLimit': True,
                'sandbox': False,  # Set to True for testing
            })

            # Test connection
            exchange.load_markets()
            print(f"Connected to {self.exchange_name} exchange")
            return exchange
        except Exception as e:
            print(f"Error initializing exchange: {e}")
            # Fallback to other exchanges
            for exchange_name in ['binance', 'coinbase', 'kraken']:
                if exchange_name != self.exchange_name:
                    try:
                        exchange_class = getattr(ccxt, exchange_name)
                        exchange = exchange_class({
                            'timeout': 30000,
                            'enableRateLimit': True,
                        })
                        exchange.load_markets()
                        print(
                            f"Connected to {exchange_name} exchange (fallback)")
                        return exchange
                    except:
                        continue

            raise Exception("Could not connect to any exchange")

    def load_model(self, feature_columns: List[str]):
        """Load the trained model"""
        self.feature_columns = feature_columns
        state_size = len(feature_columns) + 3

        self.trader = MultiprocessingCryptoRLTrader(
            state_size, action_size=3, config=self.config)
        self.trader.load_model(self.model_path)
        print(f"Loaded trained model from {self.model_path}")

    def fetch_live_data(self, timeframe='1m', limit=2) -> pd.DataFrame:
        """Fetch the latest OHLCV data using CCXT"""
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, timeframe, limit=limit)

            if not ohlcv:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

            # Convert timestamp to datetime
            df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Drop timestamp column and reorder
            df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

            return df.tail(1)  # Return only the latest complete candle

        except Exception as e:
            print(f"Error fetching live data: {e}")
            return None

    def fetch_historical_data(self, timeframe='1m', days=1) -> pd.DataFrame:
        """Fetch historical data for warmup"""
        try:
            # Calculate the start timestamp
            since = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)

            # Fetch historical data
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, timeframe, since=since)

            if not ohlcv:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

            # Convert timestamp to datetime
            df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Drop timestamp column and reorder
            df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

            return df

        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None

    def update_data_buffer(self, new_data: pd.DataFrame):
        """Update the data buffer with new data"""
        for _, row in new_data.iterrows():
            self.data_buffer.append(row.to_dict())

    def get_current_state(self) -> np.ndarray:
        """Get current market state for prediction"""
        if len(self.data_buffer) < 20:  # Need minimum data for indicators
            return None

        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.data_buffer))

        # Engineer features
        processed_data = self.feature_engineer.engineer_features(df)

        if len(processed_data) == 0:
            return None

        # Get latest row features and normalize
        latest_features_df = processed_data[self.feature_columns].iloc[-1:]
        normalized_features = self.feature_engineer.normalize_features(
            latest_features_df, self.feature_columns
        )[0]

        # Add position state
        position_state = np.array([
            self.position,
            self.balance / self.config.initial_balance,
            0  # Steps since entry (simplified for real-time)
        ])

        return np.concatenate([normalized_features, position_state])

    def execute_trade(self, action: int, current_price: float):
        """Execute trading action"""
        if action == 1 and self.position == 0:  # Buy signal
            self.position = 1
            self.entry_price = current_price
            transaction_cost = current_price * self.config.transaction_cost
            print(
                f"BUY: {self.symbol} at ${current_price:.4f} (Cost: ${transaction_cost:.2f})")

        elif action == 2 and self.position == 1:  # Sell signal
            profit_pct = (current_price - self.entry_price) / self.entry_price
            profit_amount = self.balance * profit_pct
            transaction_cost = current_price * self.config.transaction_cost

            self.balance = self.balance * \
                (1 + profit_pct - self.config.transaction_cost)

            trade_info = {
                'timestamp': datetime.now(),
                'action': 'SELL',
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'profit_pct': profit_pct,
                'profit_amount': profit_amount,
                'balance': self.balance
            }

            self.trade_history.append(trade_info)

            print(f"SELL: {self.symbol} at ${current_price:.4f}")
            print(f"Profit: {profit_pct:.2%} (${profit_amount:.2f})")
            print(f"New Balance: ${self.balance:.2f}")

            self.position = 0
            self.entry_price = 0

    def check_risk_management(self, current_price: float):
        """Check stop loss and profit target"""
        if self.position == 1:
            profit_pct = (current_price - self.entry_price) / self.entry_price

            # Stop loss check
            if profit_pct <= -self.config.stop_loss:
                print(f"STOP LOSS triggered at {profit_pct:.2%}")
                self.execute_trade(2, current_price)  # Force sell

            # Profit target check
            elif profit_pct >= self.config.profit_target_max:
                print(f"PROFIT TARGET reached at {profit_pct:.2%}")
                self.execute_trade(2, current_price)  # Force sell

    def run_live_trading(self, duration_minutes: int = 60):
        """Run live trading for specified duration"""
        print(f"Starting live trading for {duration_minutes} minutes...")
        print(f"Initial balance: ${self.balance:.2f}")

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        last_minute = None

        while datetime.now() < end_time:
            try:
                current_minute = datetime.now().replace(second=0, microsecond=0)

                # Only process new minutes
                if current_minute != last_minute:
                    # Fetch latest data
                    new_data = self.fetch_live_data()

                    if new_data is not None:
                        current_price = new_data['Close'].iloc[-1]
                        self.update_data_buffer(new_data)

                        # Check risk management first
                        self.check_risk_management(current_price)

                        # Get prediction if enough data
                        state = self.get_current_state()

                        if state is not None:
                            action = self.trader.get_action(
                                state, training=False)
                            action_names = ['HOLD', 'BUY', 'SELL']

                            print(f"[{current_minute}] Price: ${current_price:.4f}, "
                                  f"Action: {action_names[action]}, "
                                  f"Position: {'LONG' if self.position == 1 else 'NONE'}")

                            # Execute trade
                            self.execute_trade(action, current_price)

                    last_minute = current_minute

                # Sleep until next minute
                time.sleep(10)  # Check every 10 seconds

            except KeyboardInterrupt:
                print("\nTrading stopped by user")
                break
            except Exception as e:
                print(f"Error in live trading: {e}")
                time.sleep(10)

        # Final summary
        final_return = (self.balance - self.config.initial_balance) / \
            self.config.initial_balance
        print(f"\n=== Trading Session Complete ===")
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Total Return: {final_return:.2%}")
        print(f"Total Trades: {len(self.trade_history)}")

        if self.trade_history:
            winning_trades = sum(
                1 for trade in self.trade_history if trade['profit_pct'] > 0)
            win_rate = winning_trades / len(self.trade_history)
            print(f"Win Rate: {win_rate:.2%}")


def fetch_crypto_data(symbol: str = "BTC/USDT", exchange_name: str = "coinbase", timeframe: str = "1m", days: int = 7) -> pd.DataFrame:
    """Fetch cryptocurrency data using CCXT"""
    try:
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'timeout': 30000,
            'enableRateLimit': True,
        })

        # Load markets
        exchange.load_markets()

        # Calculate the start timestamp
        since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)

        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)

        if not ohlcv:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

        # Convert timestamp to datetime
        df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Drop timestamp column and reorder
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

        print(f"Fetched {len(df)} data points from {exchange_name}")
        return df

    except Exception as e:
        print(f"Error fetching data from {exchange_name}: {e}")

        # Try alternative exchanges
        alternative_exchanges = ['binance', 'coinbase', 'kraken', 'bitfinex']
        for alt_exchange in alternative_exchanges:
            if alt_exchange != exchange_name:
                try:
                    print(f"Trying {alt_exchange}...")
                    return fetch_crypto_data(symbol, alt_exchange, timeframe, days)
                except:
                    continue

        return None


def load_user_data(file_path: str) -> pd.DataFrame:
    """Load user-provided CSV data"""
    try:
        df = pd.read_csv(file_path)

        # Try to standardize column names
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'volume': 'Volume', 'timestamp': 'Datetime', 'datetime': 'Datetime',
            'time': 'Datetime', 'date': 'Datetime'
        }

        df.columns = [column_mapping.get(col.lower(), col)
                      for col in df.columns]

        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns. Found: {df.columns.tolist()}")
            print(f"Required: {required_columns}")
            return None

        # Convert datetime if exists
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])

        print(f"Loaded {len(df)} data points from {file_path}")
        return df

    except Exception as e:
        print(f"Error loading user data: {e}")
        return None


def plot_results(trader: MultiprocessingCryptoRLTrader, results: Dict):
    """Plot training and evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training losses
    if trader.training_losses:
        axes[0, 0].plot(trader.training_losses)
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')

    # Moving average of losses
    if len(trader.training_losses) > 100:
        window = 100
        moving_avg = pd.Series(trader.training_losses).rolling(
            window=window).mean()
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(
            f'Training Loss (Moving Average, window={window})')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')

    # Portfolio performance
    if results.get('trades'):
        balances = [10000]  # Initial balance
        for trade in results['trades']:
            balances.append(trade['balance'])

        axes[1, 0].plot(balances)
        axes[1, 0].set_title('Portfolio Balance Over Time')
        axes[1, 0].set_xlabel('Trade')
        axes[1, 0].set_ylabel('Balance ($)')

    # Performance metrics
    metrics = [
        f"Final Balance: ${results['final_balance']:.2f}",
        f"Total Return: {results['return_pct']:.2%}",
        f"Win Rate: {results['win_rate']:.2%}",
        f"Max Drawdown: {results['max_drawdown']:.2%}",
        f"Total Trades: {results['total_trades']}"
    ]

    axes[1, 1].text(0.1, 0.9, '\n'.join(metrics), transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='top')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def debug_dimensions(processed_data: pd.DataFrame, feature_columns: List[str]):
    """Debug function to check dimensions"""
    print(f"\n=== DIMENSION DEBUG ===")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Processed data columns: {processed_data.columns.tolist()}")
    print(f"Feature columns count: {len(feature_columns)}")
    print(f"Feature columns: {feature_columns}")

    # Create sample feature dataframe like in training
    normalized_features = np.random.randn(
        len(processed_data), len(feature_columns))
    feature_df = pd.DataFrame(normalized_features, columns=feature_columns)
    feature_df['Close'] = processed_data['Close'].values

    print(f"Feature dataframe shape: {feature_df.shape}")
    print(f"Feature dataframe columns: {feature_df.columns.tolist()}")

    # Check state dimensions
    sample_row = feature_df.iloc[0].drop(['Close'])
    print(f"Market state size (excluding Close): {len(sample_row)}")
    print(f"Expected state size: {len(sample_row) + 3}")
    print("======================\n")


def main():
    """Main function to run the crypto trading RL model"""
    # Configuration
    config = TradingConfig()
    # Use available cores minus 1
    config.num_workers = min(4, max(1, os.cpu_count() - 1))

    print(f"Using {config.num_workers} worker processes")

    # Fetch data
    print("Fetching cryptocurrency data...")
    raw_data = fetch_crypto_data("BTC/USDT", "coinbase", "1m", 7)

    if raw_data is None:
        print("Failed to fetch data. Exiting.")
        return

    # Feature engineering
    print("Engineering features...")
    feature_engineer = FeatureEngineer()
    processed_data = feature_engineer.engineer_features(raw_data)

    # Select feature columns (exclude timestamp and target columns)
    feature_columns = [col for col in processed_data.columns
                       if col not in ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Normalize features
    normalized_features = feature_engineer.normalize_features(
        processed_data, feature_columns)

    # Create feature dataframe
    feature_df = pd.DataFrame(normalized_features, columns=feature_columns)
    # Keep close price for environment
    feature_df['Close'] = processed_data['Close'].values

    # Split data
    train_size = int(0.8 * len(feature_df))
    train_data = feature_df[:train_size]
    test_data = feature_df[train_size:]

    print(f"Training data: {len(train_data)} samples")
    print(f"Testing data: {len(test_data)} samples")

    # Create test environment
    test_env = TradingEnvironment(test_data, config)

    # Initialize multiprocessing trader
    # +3 for position, balance ratio, steps since entry
    state_size = len(feature_columns) + 3
    trader = MultiprocessingCryptoRLTrader(
        state_size, action_size=3, config=config)

    # Train the model
    print("Starting multiprocessing training...")
    trader.train(train_data, episodes=1000, save_path="crypto_rl_model_mp.pth")

    # Evaluate the model
    print("Evaluating model...")
    results = trader.evaluate(test_env)

    print(f"\nEvaluation Results:")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Total Return: {results['return_pct']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Total Trades: {results['total_trades']}")

    # Plot results
    plot_results(trader, results)

    # Demonstrate continuous learning
    print("\nDemonstrating continuous learning...")
    new_data = fetch_crypto_data("BTC/USDT", "coinbase", "1m", 1)
    if new_data is not None:
        new_processed = feature_engineer.engineer_features(new_data)
        new_normalized = feature_engineer.normalize_features(
            new_processed, feature_columns)
        new_feature_df = pd.DataFrame(new_normalized, columns=feature_columns)
        new_feature_df['Close'] = new_processed['Close'].values

        trader.continuous_learning_update(new_feature_df)
        print("Model updated with new data!")


# Training script function
def train_model(user_data_path: str = None, episodes: int = 1000, model_save_path: str = "crypto_rl_model_mp.pth", num_workers: int = None):
    """Dedicated training function with multiprocessing"""
    print("="*60)
    print("MULTIPROCESSING CRYPTOCURRENCY RL TRADING MODEL - TRAINING MODE")
    print("="*60)

    # Configuration
    config = TradingConfig()
    if num_workers is None:
        # Max out workers but leave 2 cores for system overhead
        total_cores = os.cpu_count()
        # Minimum 2 workers, leave 2 cores for overhead
        config.num_workers = max(2, total_cores - 2)
    else:
        config.num_workers = num_workers

    print(
        f"System cores: {os.cpu_count()}, Using {config.num_workers} worker processes")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load data - user provided or default
    if user_data_path:
        print(f"Loading user data from: {user_data_path}")
        raw_data = load_user_data(user_data_path)
        if raw_data is None:
            print("Failed to load user data. Falling back to default data source.")
            raw_data = fetch_crypto_data("BTC/USDT", "coinbase", "1m", 7)
    else:
        print("Fetching default cryptocurrency data...")
        raw_data = fetch_crypto_data("BTC/USDT", "coinbase", "1m", 7)

    if raw_data is None:
        print("Failed to fetch data. Exiting.")
        return None

    # Feature engineering
    print("Engineering features...")
    feature_engineer = FeatureEngineer()
    processed_data = feature_engineer.engineer_features(raw_data)

    # Select feature columns
    feature_columns = [col for col in processed_data.columns
                       if col not in ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Debug dimensions
    debug_dimensions(processed_data, feature_columns)

    # Normalize features
    normalized_features = feature_engineer.normalize_features(
        processed_data, feature_columns)

    # Create feature dataframe
    feature_df = pd.DataFrame(normalized_features, columns=feature_columns)
    feature_df['Close'] = processed_data['Close'].values

    # Split data
    train_size = int(0.8 * len(feature_df))
    train_data = feature_df[:train_size]
    test_data = feature_df[train_size:]

    print(f"Training data: {len(train_data)} samples")
    print(f"Testing data: {len(test_data)} samples")

    # Create environments
    test_env = TradingEnvironment(test_data, config)

    # Initialize multiprocessing trader
    state_size = len(feature_columns) + 3
    trader = MultiprocessingCryptoRLTrader(
        state_size, action_size=3, config=config)

    # Train the model
    print(f"Starting multiprocessing training for {episodes} episodes...")
    start_time = time.time()

    try:
        trader.train(train_data, episodes=episodes, save_path=model_save_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        trader.save_model(model_save_path.replace('.pth', '_interrupted.pth'))
        return None

    training_time = time.time() - start_time

    # Evaluate the model
    print("Evaluating model...")
    results = trader.evaluate(test_env)

    print(f"\n" + "="*60)
    print("MULTIPROCESSING TRAINING COMPLETE - EVALUATION RESULTS:")
    print("="*60)
    print(f"Training Time: {training_time:.1f} seconds")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Total Return: {results['return_pct']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Training Steps: {len(trader.training_losses)}")

    # Save feature columns for later use
    feature_columns_path = model_save_path.replace('.pth', '_features.txt')
    with open(feature_columns_path, 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")

    print(f"\nModel saved to: {model_save_path}")
    print(f"Feature columns saved to: {feature_columns_path}")

    # Plot results
    plot_results(trader, results)

    return trader, feature_columns


# Live trading script function
def run_live_trading(model_path: str = "crypto_rl_model_mp.pth",
                     symbol: str = "BTC/USDT",
                     exchange: str = "coinbase",
                     duration_minutes: int = 60):
    """Run live trading with trained model"""
    print("="*60)
    print("MULTIPROCESSING CRYPTOCURRENCY RL TRADING MODEL - LIVE TRADING MODE")
    print("="*60)

    # Load feature columns
    feature_columns_path = model_path.replace('.pth', '_features.txt')
    try:
        with open(feature_columns_path, 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(feature_columns)} feature columns")
    except FileNotFoundError:
        print(f"Feature columns file not found: {feature_columns_path}")
        print("Please train the model first.")
        return

    # Initialize real-time trader
    config = TradingConfig()
    rt_trader = RealTimeTrader(model_path, symbol, exchange, config)
    rt_trader.load_model(feature_columns)

    # Warm up with historical data
    print("Warming up with historical data...")
    warmup_data = rt_trader.fetch_historical_data("1m", 1)
    if warmup_data is not None:
        rt_trader.update_data_buffer(warmup_data)
        print(f"Loaded {len(warmup_data)} minutes of warmup data")

    # Start live trading
    rt_trader.run_live_trading(duration_minutes)


if __name__ == "__main__":
    import sys

    # Ensure proper multiprocessing behavior
    mp.freeze_support()

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  Training: python crypto_rl_trader.py train [data_file.csv] [episodes] [model_file.pth] [num_workers]")
        print(
            "  Live Trading: python crypto_rl_trader.py trade [model_file.pth] [symbol] [exchange] [duration_minutes]")
        print("  Example: python crypto_rl_trader.py train my_data.csv 2000 model.pth 6")
        print("  Example: python crypto_rl_trader.py trade crypto_rl_model_mp.pth BTC/USDT coinbase 120")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "train":
        data_file = sys.argv[2] if len(
            sys.argv) > 2 and sys.argv[2] != "default" else None
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        model_path = sys.argv[4] if len(
            sys.argv) > 4 else "crypto_rl_model_mp.pth"
        num_workers = int(sys.argv[5]) if len(sys.argv) > 5 else None

        train_model(data_file, episodes, model_path, num_workers)

    elif mode == "trade":
        model_path = sys.argv[2] if len(
            sys.argv) > 2 else "crypto_rl_model_mp.pth"
        symbol = sys.argv[3] if len(sys.argv) > 3 else "BTC/USDT"
        exchange = sys.argv[4] if len(sys.argv) > 4 else "coinbase"
        duration = int(sys.argv[5]) if len(sys.argv) > 5 else 60

        run_live_trading(model_path, symbol, exchange, duration)

    else:
        print(f"Unknown mode: {mode}. Use 'train' or 'trade'.")
