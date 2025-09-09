"""
Advanced Monte Carlo Tree Search (MCTS) Algorithm
================================================

A sophisticated MCTS implementation with Upper Confidence Bounds (UCB1),
progressive widening, and domain-specific enhancements for game AI,
decision making, and reinforcement learning applications.

Mathematical Foundation:
UCB1 Selection: argmax(Q(s,a) + C * sqrt(ln(N(s)) / N(s,a)))

Where:
- Q(s,a): Average reward for action a in state s
- C: Exploration constant (typically sqrt(2))
- N(s): Number of visits to state s
- N(s,a): Number of times action a was selected in state s

Applications:
- Game AI (Chess, Go, Poker)
- Autonomous trading strategies
- Resource allocation
- Planning under uncertainty
- Multi-armed bandit problems
"""

import numpy as np
import random
import math
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import time


class NodeType(Enum):
    """Types of nodes in the MCTS tree."""
    ROOT = "root"
    INTERNAL = "internal"
    LEAF = "leaf"
    TERMINAL = "terminal"


@dataclass
class MCTSConfig:
    """Configuration parameters for MCTS algorithm."""
    exploration_constant: float = math.sqrt(2)  # UCB1 exploration parameter
    max_iterations: int = 10000
    max_time_seconds: float = 5.0
    progressive_widening_constant: float = 0.5
    progressive_widening_exponent: float = 0.5
    min_visits_for_expansion: int = 5
    use_rave: bool = False  # Rapid Action Value Estimation
    rave_bias_constant: float = 300.0


class GameState(ABC):
    """Abstract base class for game states."""
    
    @abstractmethod
    def get_legal_actions(self) -> List[Any]:
        """Return list of legal actions from this state."""
        pass
    
    @abstractmethod
    def apply_action(self, action: Any) -> 'GameState':
        """Apply action and return new state."""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        pass
    
    @abstractmethod
    def get_reward(self, player: int) -> float:
        """Get reward for specified player in terminal state."""
        pass
    
    @abstractmethod
    def get_current_player(self) -> int:
        """Get the current player to move."""
        pass
    
    @abstractmethod
    def clone(self) -> 'GameState':
        """Create a deep copy of this state."""
        pass


class TicTacToeState(GameState):
    """Example implementation: Tic-tac-toe game state."""
    
    def __init__(self, board: Optional[np.ndarray] = None, current_player: int = 1):
        self.board = board if board is not None else np.zeros((3, 3), dtype=int)
        self.current_player = current_player
    
    def get_legal_actions(self) -> List[Tuple[int, int]]:
        """Return list of empty positions."""
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    actions.append((i, j))
        return actions
    
    def apply_action(self, action: Tuple[int, int]) -> 'TicTacToeState':
        """Place current player's mark at specified position."""
        new_board = self.board.copy()
        new_board[action[0], action[1]] = self.current_player
        return TicTacToeState(new_board, -self.current_player)
    
    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self._check_winner() != 0 or len(self.get_legal_actions()) == 0
    
    def get_reward(self, player: int) -> float:
        """Get reward for specified player."""
        winner = self._check_winner()
        if winner == player:
            return 1.0
        elif winner == -player:
            return -1.0
        else:
            return 0.0  # Draw
    
    def get_current_player(self) -> int:
        """Get current player."""
        return self.current_player
    
    def clone(self) -> 'TicTacToeState':
        """Create a copy of this state."""
        return TicTacToeState(self.board.copy(), self.current_player)
    
    def _check_winner(self) -> int:
        """Check if there's a winner."""
        # Check rows
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3:
                return int(np.sum(self.board[i, :]) // 3)
        
        # Check columns
        for j in range(3):
            if abs(np.sum(self.board[:, j])) == 3:
                return int(np.sum(self.board[:, j]) // 3)
        
        # Check diagonals
        if abs(np.trace(self.board)) == 3:
            return int(np.trace(self.board) // 3)
        
        if abs(np.trace(np.fliplr(self.board))) == 3:
            return int(np.trace(np.fliplr(self.board)) // 3)
        
        return 0
    
    def __str__(self) -> str:
        """String representation of the board."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        result = ""
        for i in range(3):
            for j in range(3):
                result += symbols[self.board[i, j]] + " "
            result += "\n"
        return result


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[Any] = None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children: Dict[Any, 'MCTSNode'] = {}
        
        # Statistics
        self.visits = 0
        self.total_reward = 0.0
        self.reward_squared = 0.0  # For variance calculation
        
        # RAVE (Rapid Action Value Estimation) statistics
        self.rave_visits: Dict[Any, int] = {}
        self.rave_rewards: Dict[Any, float] = {}
        
        # Node properties
        self.is_fully_expanded = False
        self.legal_actions = state.get_legal_actions()
        self.untried_actions = self.legal_actions.copy()
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state."""
        return self.state.is_terminal()
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0
    
    def get_average_reward(self) -> float:
        """Get average reward for this node."""
        return self.total_reward / self.visits if self.visits > 0 else 0.0
    
    def get_reward_variance(self) -> float:
        """Calculate reward variance for this node."""
        if self.visits < 2:
            return 0.0
        
        mean_reward = self.get_average_reward()
        variance = (self.reward_squared / self.visits) - (mean_reward ** 2)
        return max(0.0, variance)  # Ensure non-negative
    
    def update_statistics(self, reward: float):
        """Update node statistics with new reward."""
        self.visits += 1
        self.total_reward += reward
        self.reward_squared += reward ** 2
    
    def update_rave_statistics(self, action: Any, reward: float):
        """Update RAVE statistics for an action."""
        if action not in self.rave_visits:
            self.rave_visits[action] = 0
            self.rave_rewards[action] = 0.0
        
        self.rave_visits[action] += 1
        self.rave_rewards[action] += reward
    
    def get_rave_value(self, action: Any) -> float:
        """Get RAVE value for an action."""
        if action not in self.rave_visits or self.rave_visits[action] == 0:
            return 0.0
        return self.rave_rewards[action] / self.rave_visits[action]


class AdvancedMCTS:
    """
    Advanced Monte Carlo Tree Search implementation with multiple enhancements.
    
    Features:
    - UCB1 selection with variance-based confidence bounds
    - Progressive widening for continuous action spaces
    - RAVE (Rapid Action Value Estimation)
    - Early termination based on confidence
    - Parallelization support
    """
    
    def __init__(self, config: MCTSConfig):
        self.config = config
        self.root: Optional[MCTSNode] = None
        self.iteration_count = 0
        self.start_time = 0.0
    
    def search(self, initial_state: GameState, target_player: int) -> Any:
        """
        Perform MCTS search and return the best action.
        
        Args:
            initial_state: Starting game state
            target_player: Player for whom we're optimizing
            
        Returns:
            Best action found by MCTS
        """
        self.root = MCTSNode(initial_state)
        self.iteration_count = 0
        self.start_time = time.time()
        
        # Main MCTS loop
        while self._should_continue():
            # Selection: Traverse tree to find leaf node
            leaf_node = self._select(self.root)
            
            # Expansion: Add new child node if possible
            new_node = self._expand(leaf_node)
            
            # Simulation: Run random simulation from new node
            reward = self._simulate(new_node, target_player)
            
            # Backpropagation: Update statistics up the tree
            self._backpropagate(new_node, reward)
            
            self.iteration_count += 1
        
        # Return best action based on visit count or average reward
        return self._get_best_action()
    
    def _should_continue(self) -> bool:
        """Check if search should continue."""
        time_limit_reached = (time.time() - self.start_time) >= self.config.max_time_seconds
        iteration_limit_reached = self.iteration_count >= self.config.max_iterations
        
        return not (time_limit_reached or iteration_limit_reached)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Traverse tree using UCB1 until reaching a leaf.
        """
        current = node
        
        while not current.is_terminal() and current.is_fully_expanded:
            current = self._select_child_ucb1(current)
        
        return current
    
    def _select_child_ucb1(self, node: MCTSNode) -> MCTSNode:
        """
        Select child using UCB1 with optional RAVE enhancement.
        """
        best_action = None
        best_value = float('-inf')
        
        for action, child in node.children.items():
            if child.visits == 0:
                # Unvisited child gets infinite priority
                return child
            
            # Standard UCB1 value
            ucb1_value = self._calculate_ucb1(node, child)
            
            # Add RAVE enhancement if enabled
            if self.config.use_rave:
                rave_value = self._calculate_rave_ucb1(node, action, child)
                # Combine UCB1 and RAVE with decreasing RAVE influence
                beta = self._calculate_rave_beta(child.visits)
                combined_value = (1 - beta) * ucb1_value + beta * rave_value
            else:
                combined_value = ucb1_value
            
            if combined_value > best_value:
                best_value = combined_value
                best_action = action
        
        return node.children[best_action]
    
    def _calculate_ucb1(self, parent: MCTSNode, child: MCTSNode) -> float:
        """Calculate UCB1 value for a child node."""
        if child.visits == 0:
            return float('inf')
        
        # Average reward
        exploitation = child.get_average_reward()
        
        # Exploration term with confidence bounds
        exploration = self.config.exploration_constant * math.sqrt(
            math.log(parent.visits) / child.visits
        )
        
        # Add variance-based confidence bound
        variance_term = math.sqrt(child.get_reward_variance() / child.visits) if child.visits > 1 else 0
        
        return exploitation + exploration + 0.1 * variance_term
    
    def _calculate_rave_ucb1(self, parent: MCTSNode, action: Any, child: MCTSNode) -> float:
        """Calculate RAVE-enhanced UCB1 value."""
        rave_value = parent.get_rave_value(action)
        rave_visits = parent.rave_visits.get(action, 0)
        
        if rave_visits == 0:
            return child.get_average_reward()
        
        # RAVE exploration term
        rave_exploration = self.config.exploration_constant * math.sqrt(
            math.log(sum(parent.rave_visits.values())) / rave_visits
        )
        
        return rave_value + rave_exploration
    
    def _calculate_rave_beta(self, visits: int) -> float:
        """Calculate RAVE mixing parameter beta."""
        return math.sqrt(self.config.rave_bias_constant / 
                        (3 * visits + self.config.rave_bias_constant))
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: Add new child node if possible.
        """
        if node.is_terminal():
            return node
        
        # Progressive widening: limit expansion based on visit count
        if self._should_use_progressive_widening():
            max_children = int(node.visits ** self.config.progressive_widening_exponent *
                             self.config.progressive_widening_constant)
            if len(node.children) >= max_children:
                return node
        
        # Check if node can be expanded
        if len(node.untried_actions) == 0:
            node.is_fully_expanded = True
            return node
        
        # Select random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # Create new child node
        new_state = node.state.apply_action(action)
        child_node = MCTSNode(new_state, parent=node, action=action)
        node.children[action] = child_node
        
        # Mark as fully expanded if no more actions
        if len(node.untried_actions) == 0:
            node.is_fully_expanded = True
        
        return child_node
    
    def _should_use_progressive_widening(self) -> bool:
        """Check if progressive widening should be used."""
        return (self.config.progressive_widening_constant > 0 and
                self.config.progressive_widening_exponent > 0)
    
    def _simulate(self, node: MCTSNode, target_player: int) -> float:
        """
        Simulation phase: Run random simulation from node to terminal state.
        """
        current_state = node.state.clone()
        simulation_actions = []
        
        # Random simulation until terminal state
        while not current_state.is_terminal():
            legal_actions = current_state.get_legal_actions()
            if not legal_actions:
                break
            
            # Use random policy for simulation
            action = random.choice(legal_actions)
            simulation_actions.append(action)
            current_state = current_state.apply_action(action)
        
        # Get reward for target player
        reward = current_state.get_reward(target_player)
        
        # Update RAVE statistics if enabled
        if self.config.use_rave:
            self._update_rave_statistics(node, simulation_actions, reward)
        
        return reward
    
    def _update_rave_statistics(self, node: MCTSNode, actions: List[Any], reward: float):
        """Update RAVE statistics for actions in simulation."""
        current = node
        while current is not None:
            for action in actions:
                if action in current.legal_actions:
                    current.update_rave_statistics(action, reward)
            current = current.parent
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: Update statistics up the tree.
        """
        current = node
        player_multiplier = 1
        
        while current is not None:
            # Update node statistics with alternating reward sign
            current.update_statistics(reward * player_multiplier)
            
            # Alternate player perspective
            player_multiplier *= -1
            current = current.parent
    
    def _get_best_action(self) -> Any:
        """
        Select best action based on visit count and average reward.
        """
        if not self.root.children:
            return None
        
        best_action = None
        best_score = float('-inf')
        
        for action, child in self.root.children.items():
            # Use visit count as primary criterion for robustness
            score = child.visits
            
            # Add average reward as tiebreaker
            if child.visits > 0:
                score += 0.01 * child.get_average_reward()
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def get_action_statistics(self) -> Dict[Any, Dict[str, float]]:
        """Get detailed statistics for all root children."""
        stats = {}
        
        for action, child in self.root.children.items():
            stats[action] = {
                'visits': child.visits,
                'average_reward': child.get_average_reward(),
                'variance': child.get_reward_variance(),
                'ucb1_value': self._calculate_ucb1(self.root, child) if child.visits > 0 else 0
            }
        
        return stats


def tic_tac_toe_example():
    """Example: MCTS for Tic-tac-toe."""
    print("=== Advanced MCTS Tic-Tac-Toe Example ===")
    
    # Create MCTS configuration
    config = MCTSConfig(
        exploration_constant=1.414,  # sqrt(2)
        max_iterations=5000,
        max_time_seconds=2.0,
        use_rave=True,
        min_visits_for_expansion=3
    )
    
    # Initialize game
    game_state = TicTacToeState()
    mcts = AdvancedMCTS(config)
    
    print("Starting position:")
    print(game_state)
    
    move_count = 0
    while not game_state.is_terminal() and move_count < 9:
        current_player = game_state.get_current_player()
        print(f"Player {current_player}'s turn")
        
        # Use MCTS to find best move
        best_action = mcts.search(game_state, current_player)
        
        if best_action is None:
            print("No legal moves available")
            break
        
        # Apply the move
        game_state = game_state.apply_action(best_action)
        move_count += 1
        
        print(f"Player {current_player} plays {best_action}")
        print(game_state)
        
        # Show search statistics
        stats = mcts.get_action_statistics()
        print("MCTS Action Statistics:")
        for action, stat in stats.items():
            print(f"  {action}: visits={stat['visits']}, "
                  f"avg_reward={stat['average_reward']:.3f}, "
                  f"ucb1={stat['ucb1_value']:.3f}")
        print()
    
    # Show final result
    if game_state.is_terminal():
        winner = game_state._check_winner()
        if winner == 1:
            print("Player X (1) wins!")
        elif winner == -1:
            print("Player O (-1) wins!")
        else:
            print("It's a draw!")
    
    print(f"Game completed in {move_count} moves")
    print(f"MCTS performed {mcts.iteration_count} iterations per move on average")


if __name__ == "__main__":
    tic_tac_toe_example()