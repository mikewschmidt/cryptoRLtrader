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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


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
    """Experience replay buffer for storing and sampling experiences"""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, td_error: float = 1.0):
        """Add experience to buffer with priority"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, np.ndarray]:
        """Sample batch with prioritized experience replay"""
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

        market_state = self.data.iloc[self.current_step].values
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


class CryptoRLTrader:
    """Main RL trader class with continuous learning capabilities"""

    def __init__(self, state_size: int, action_size: int = 3, config: TradingConfig = None):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or TradingConfig()

        # Neural networks
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        # Replay buffer
        self.memory = ReplayBuffer(capacity=100000)

        # Training parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.update_target_freq = 1000
        self.learn_freq = 4
        self.step_count = 0

        # Performance tracking
        self.training_rewards = []
        self.training_losses = []
        self.performance_history = []

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Get action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        # Calculate TD error for prioritized replay
        with torch.no_grad():
            state_tensor = torch.FloatTensor(
                state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(
                next_state).unsqueeze(0).to(self.device)

            current_q = self.q_network(state_tensor)[0][action]
            if done:
                target_q = reward
            else:
                next_q = self.target_network(next_state_tensor).max(1)[0]
                target_q = reward + 0.99 * next_q

            td_error = abs(current_q - target_q).item()

        self.memory.add(state, action, reward, next_state, done, td_error)

    def replay(self) -> float:
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0

        # Sample batch from replay buffer
        experiences, weights = self.memory.sample(self.batch_size)

        states = torch.FloatTensor([e[0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor(
            [e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(
            [e[3] for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in experiences]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)

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

        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return weighted_loss.item()

    def train(self, env: TradingEnvironment, episodes: int, save_path: str = "crypto_rl_model.pth"):
        """Train the RL agent"""
        best_reward = float('-inf')

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0

            while True:
                action = self.get_action(state, training=True)
                next_state, reward, done, info = env.step(action)

                self.remember(state, action, reward, next_state, done)
                episode_reward += reward

                # Train the model
                if self.step_count % self.learn_freq == 0:
                    loss = self.replay()
                    episode_loss += loss

                # Update target network
                if self.step_count % self.update_target_freq == 0:
                    self.update_target_network()

                state = next_state
                self.step_count += 1
                steps += 1

                if done:
                    break

            self.training_rewards.append(episode_reward)
            self.training_losses.append(
                episode_loss / steps if steps > 0 else 0)

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'q_network_state_dict': self.q_network.state_dict(),
                    'target_network_state_dict': self.target_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon,
                    'step_count': self.step_count
                }, save_path)

            if episode % 100 == 0:
                avg_reward = np.mean(self.training_rewards[-100:])
                avg_loss = np.mean(self.training_losses[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}, Balance: {env.balance:.2f}")

    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(
            checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.step_count = checkpoint.get('step_count', 0)

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

    def __init__(self, model_path: str, symbol: str = "BTC/USDT", exchange: str = "binance", config: TradingConfig = None):
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

        self.trader = CryptoRLTrader(
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

        # Get latest row features
        latest_features = processed_data[self.feature_columns].iloc[-1].values
        normalized_features = self.feature_engineer.normalize_features(
            processed_data[self.feature_columns].iloc[-1:], self.feature_columns
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


def fetch_crypto_data(symbol: str = "BTC/USDT", exchange_name: str = "binance", timeframe: str = "1m", days: int = 7) -> pd.DataFrame:
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


def plot_results(trader: CryptoRLTrader, results: Dict):
    """Plot training and evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training rewards
    axes[0, 0].plot(trader.training_rewards)
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')

    # Training losses
    axes[0, 1].plot(trader.training_losses)
    axes[0, 1].set_title('Training Losses')
    axes[0, 1].set_xlabel('Episode')
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


def main():
    """Main function to run the crypto trading RL model"""
    # Configuration
    config = TradingConfig()

    # Fetch data
    print("Fetching cryptocurrency data...")
    raw_data = fetch_crypto_data("BTC/USDT", "binance", "1m", 7)

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

    # Create environments
    train_env = TradingEnvironment(train_data, config)
    test_env = TradingEnvironment(test_data, config)

    # Initialize trader
    # +3 for position, balance ratio, steps since entry
    state_size = len(feature_columns) + 3
    trader = CryptoRLTrader(state_size, action_size=3, config=config)

    # Train the model
    print("Starting training...")
    trader.train(train_env, episodes=1000, save_path="crypto_rl_model.pth")

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
    new_data = fetch_crypto_data("BTC/USDT", "binance", "1m", 1)
    if new_data is not None:
        new_processed = feature_engineer.engineer_features(new_data)
        new_normalized = feature_engineer.normalize_features(
            new_processed, feature_columns)
        new_feature_df = pd.DataFrame(new_normalized, columns=feature_columns)
        new_feature_df['Close'] = new_processed['Close'].values

        trader.continuous_learning_update(new_feature_df)
        print("Model updated with new data!")


# Training script function
def train_model(user_data_path: str = None, episodes: int = 1000, model_save_path: str = "crypto_rl_model.pth"):
    """Dedicated training function"""
    print("="*50)
    print("CRYPTOCURRENCY RL TRADING MODEL - TRAINING MODE")
    print("="*50)

    # Configuration
    config = TradingConfig()

    # Load data - user provided or default
    if user_data_path:
        print(f"Loading user data from: {user_data_path}")
        raw_data = load_user_data(user_data_path)
        if raw_data is None:
            print("Failed to load user data. Falling back to default data source.")
            raw_data = fetch_crypto_data("BTC/USDT", "binance", "1m", 7)
    else:
        print("Fetching default cryptocurrency data...")
        raw_data = fetch_crypto_data("BTC/USDT", "binance", "1m", 7)

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
    train_env = TradingEnvironment(train_data, config)
    test_env = TradingEnvironment(test_data, config)

    # Initialize trader
    state_size = len(feature_columns) + 3
    trader = CryptoRLTrader(state_size, action_size=3, config=config)

    # Train the model
    print(f"Starting training for {episodes} episodes...")
    trader.train(train_env, episodes=episodes, save_path=model_save_path)

    # Evaluate the model
    print("Evaluating model...")
    results = trader.evaluate(test_env)

    print(f"\n" + "="*50)
    print("TRAINING COMPLETE - EVALUATION RESULTS:")
    print("="*50)
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Total Return: {results['return_pct']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Total Trades: {results['total_trades']}")

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
def run_live_trading(model_path: str = "crypto_rl_model.pth",
                     symbol: str = "BTC/USDT",
                     exchange: str = "binance",
                     duration_minutes: int = 60):
    """Run live trading with trained model"""
    print("="*50)
    print("CRYPTOCURRENCY RL TRADING MODEL - LIVE TRADING MODE")
    print("="*50)

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

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  Training: python crypto_rl_trader.py train [data_file.csv] [episodes] [model_file.pth]")
        print(
            "  Live Trading: python crypto_rl_trader.py trade [model_file.pth] [symbol] [exchange] [duration_minutes]")
        print("  Example: python crypto_rl_trader.py train my_data.csv 1500")
        print("  Example: python crypto_rl_trader.py trade crypto_rl_model.pth BTC/USDT binance 120")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "train":
        data_file = sys.argv[2] if len(
            sys.argv) > 2 and sys.argv[2] != "default" else None
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        model_path = sys.argv[4] if len(
            sys.argv) > 4 else "crypto_rl_model.pth"

        train_model(data_file, episodes, model_path)

    elif mode == "trade":
        model_path = sys.argv[2] if len(
            sys.argv) > 2 else "crypto_rl_model.pth"
        symbol = sys.argv[3] if len(sys.argv) > 3 else "BTC/USDT"
        exchange = sys.argv[4] if len(sys.argv) > 4 else "binance"
        duration = int(sys.argv[5]) if len(sys.argv) > 5 else 60

        run_live_trading(model_path, symbol, exchange, duration)

    else:
        print(f"Unknown mode: {mode}. Use 'train' or 'trade'.")
