/**
 * Advanced Monte Carlo Tree Search (MCTS) Implementation
 * JavaScript version of the Python algorithm
 * 
 * Features:
 * - UCB1 selection strategy
 * - Progressive widening
 * - Configurable exploration parameters
 * - Game-agnostic implementation
 */

class GameState {
    /**
     * Abstract base class for game states
     */
    getLegalActions() {
        throw new Error('getLegalActions must be implemented');
    }
    
    applyAction(action) {
        throw new Error('applyAction must be implemented');
    }
    
    isTerminal() {
        throw new Error('isTerminal must be implemented');
    }
    
    getReward(player) {
        throw new Error('getReward must be implemented');
    }
    
    getCurrentPlayer() {
        throw new Error('getCurrentPlayer must be implemented');
    }
    
    clone() {
        throw new Error('clone must be implemented');
    }
}

class TicTacToeState extends GameState {
    constructor(board = null, currentPlayer = 1) {
        super();
        this.board = board || Array(3).fill().map(() => Array(3).fill(0));
        this.currentPlayer = currentPlayer;
    }
    
    getLegalActions() {
        const actions = [];
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                if (this.board[i][j] === 0) {
                    actions.push([i, j]);
                }
            }
        }
        return actions;
    }
    
    applyAction(action) {
        const [row, col] = action;
        const newBoard = this.board.map(r => [...r]);
        newBoard[row][col] = this.currentPlayer;
        return new TicTacToeState(newBoard, -this.currentPlayer);
    }
    
    isTerminal() {
        return this._checkWinner() !== 0 || this.getLegalActions().length === 0;
    }
    
    getReward(player) {
        const winner = this._checkWinner();
        if (winner === player) return 1.0;
        if (winner === -player) return -1.0;
        return 0.0;
    }
    
    getCurrentPlayer() {
        return this.currentPlayer;
    }
    
    clone() {
        return new TicTacToeState(
            this.board.map(r => [...r]), 
            this.currentPlayer
        );
    }
    
    _checkWinner() {
        // Check rows
        for (let i = 0; i < 3; i++) {
            const sum = this.board[i].reduce((a, b) => a + b, 0);
            if (Math.abs(sum) === 3) return Math.sign(sum);
        }
        
        // Check columns
        for (let j = 0; j < 3; j++) {
            const sum = this.board.map(row => row[j]).reduce((a, b) => a + b, 0);
            if (Math.abs(sum) === 3) return Math.sign(sum);
        }
        
        // Check diagonals
        const diag1 = this.board[0][0] + this.board[1][1] + this.board[2][2];
        if (Math.abs(diag1) === 3) return Math.sign(diag1);
        
        const diag2 = this.board[0][2] + this.board[1][1] + this.board[2][0];
        if (Math.abs(diag2) === 3) return Math.sign(diag2);
        
        return 0;
    }
    
    toString() {
        const symbols = {0: '.', 1: 'X', '-1': 'O'};
        return this.board.map(row => 
            row.map(cell => symbols[cell]).join(' ')
        ).join('\\n');
    }
}

class MCTSNode {
    constructor(state, parent = null, action = null) {
        this.state = state;
        this.parent = parent;
        this.action = action;
        this.children = new Map();
        
        // Statistics
        this.visits = 0;
        this.totalReward = 0.0;
        this.rewardSquared = 0.0;
        
        // Node properties
        this.isFullyExpanded = false;
        this.legalActions = state.getLegalActions();
        this.untriedActions = [...this.legalActions];
    }
    
    isTerminal() {
        return this.state.isTerminal();
    }
    
    isLeaf() {
        return this.children.size === 0;
    }
    
    getAverageReward() {
        return this.visits > 0 ? this.totalReward / this.visits : 0.0;
    }
    
    getRewardVariance() {
        if (this.visits < 2) return 0.0;
        const meanReward = this.getAverageReward();
        const variance = (this.rewardSquared / this.visits) - (meanReward ** 2);
        return Math.max(0.0, variance);
    }
    
    updateStatistics(reward) {
        this.visits++;
        this.totalReward += reward;
        this.rewardSquared += reward ** 2;
    }
}

class AdvancedMCTS {
    constructor(config = {}) {
        this.config = {
            explorationConstant: Math.sqrt(2),
            maxIterations: 10000,
            maxTimeSeconds: 5.0,
            progressiveWidening: false,
            progressiveWideningConstant: 0.5,
            progressiveWideningExponent: 0.5,
            minVisitsForExpansion: 5,
            ...config
        };
        
        this.root = null;
        this.iterationCount = 0;
        this.startTime = 0;
    }
    
    search(initialState, targetPlayer) {
        this.root = new MCTSNode(initialState);
        this.iterationCount = 0;
        this.startTime = Date.now();
        
        // Main MCTS loop
        while (this._shouldContinue()) {
            // Selection: Traverse tree to find leaf node
            const leafNode = this._select(this.root);
            
            // Expansion: Add new child node if possible
            const newNode = this._expand(leafNode);
            
            // Simulation: Run random simulation from new node
            const reward = this._simulate(newNode, targetPlayer);
            
            // Backpropagation: Update statistics up the tree
            this._backpropagate(newNode, reward);
            
            this.iterationCount++;
        }
        
        // Return best action based on visit count
        return this._getBestAction();
    }
    
    _shouldContinue() {
        const timeLimit = (Date.now() - this.startTime) >= (this.config.maxTimeSeconds * 1000);
        const iterationLimit = this.iterationCount >= this.config.maxIterations;
        return !timeLimit && !iterationLimit;
    }
    
    _select(node) {
        let current = node;
        
        while (!current.isTerminal() && current.isFullyExpanded) {
            current = this._selectChildUCB1(current);
        }
        
        return current;
    }
    
    _selectChildUCB1(node) {
        let bestAction = null;
        let bestValue = -Infinity;
        
        for (const [action, child] of node.children) {
            if (child.visits === 0) {
                return child; // Unvisited child gets infinite priority
            }
            
            const ucb1Value = this._calculateUCB1(node, child);
            
            if (ucb1Value > bestValue) {
                bestValue = ucb1Value;
                bestAction = action;
            }
        }
        
        return node.children.get(bestAction);
    }
    
    _calculateUCB1(parent, child) {
        if (child.visits === 0) return Infinity;
        
        // Average reward
        const exploitation = child.getAverageReward();
        
        // Exploration term
        const exploration = this.config.explorationConstant * 
            Math.sqrt(Math.log(parent.visits) / child.visits);
        
        // Variance-based confidence bound
        const varianceTerm = child.visits > 1 ? 
            Math.sqrt(child.getRewardVariance() / child.visits) : 0;
        
        return exploitation + exploration + 0.1 * varianceTerm;
    }
    
    _expand(node) {
        if (node.isTerminal()) {
            return node;
        }
        
        // Progressive widening check
        if (this.config.progressiveWidening) {
            const maxChildren = Math.floor(
                node.visits ** this.config.progressiveWideningExponent *
                this.config.progressiveWidening 
            );
            if (node.children.size >= maxChildren) {
                return node;
            }
        }
        
        // Check if node can be expanded
        if (node.untriedActions.length === 0) {
            node.isFullyExpanded = true;
            return node;
        }
        
        // Select random untried action
        const actionIndex = Math.floor(Math.random() * node.untriedActions.length);
        const action = node.untriedActions[actionIndex];
        node.untriedActions.splice(actionIndex, 1);
        
        // Create new child node
        const newState = node.state.applyAction(action);
        const childNode = new MCTSNode(newState, node, action);
        node.children.set(JSON.stringify(action), childNode);
        
        // Mark as fully expanded if no more actions
        if (node.untriedActions.length === 0) {
            node.isFullyExpanded = true;
        }
        
        return childNode;
    }
    
    _simulate(node, targetPlayer) {
        let currentState = node.state.clone();
        
        // Random simulation until terminal state
        while (!currentState.isTerminal()) {
            const legalActions = currentState.getLegalActions();
            if (legalActions.length === 0) break;
            
            // Use random policy for simulation
            const randomAction = legalActions[Math.floor(Math.random() * legalActions.length)];
            currentState = currentState.applyAction(randomAction);
        }
        
        // Get reward for target player
        return currentState.getReward(targetPlayer);
    }
    
    _backpropagate(node, reward) {
        let current = node;
        let playerMultiplier = 1;
        
        while (current !== null) {
            // Update node statistics with alternating reward sign
            current.updateStatistics(reward * playerMultiplier);
            
            // Alternate player perspective
            playerMultiplier *= -1;
            current = current.parent;
        }
    }
    
    _getBestAction() {
        if (this.root.children.size === 0) {
            return null;
        }
        
        let bestAction = null;
        let bestScore = -Infinity;
        
        for (const [actionStr, child] of this.root.children) {
            // Use visit count as primary criterion
            let score = child.visits;
            
            // Add average reward as tiebreaker
            if (child.visits > 0) {
                score += 0.01 * child.getAverageReward();
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestAction = JSON.parse(actionStr);
            }
        }
        
        return bestAction;
    }
    
    getActionStatistics() {
        const stats = {};
        
        for (const [actionStr, child] of this.root.children) {
            const action = JSON.parse(actionStr);
            stats[actionStr] = {
                visits: child.visits,
                averageReward: child.getAverageReward(),
                variance: child.getRewardVariance(),
                ucb1Value: child.visits > 0 ? this._calculateUCB1(this.root, child) : 0
            };
        }
        
        return stats;
    }
}

// Example usage and testing
function runTicTacToeExample() {
    console.log('=== Advanced MCTS Tic-Tac-Toe Example (JavaScript) ===');
    
    // Create MCTS configuration
    const config = {
        explorationConstant: Math.sqrt(2),
        maxIterations: 5000,
        maxTimeSeconds: 2.0,
        progressiveWidening: false
    };
    
    // Initialize game
    let gameState = new TicTacToeState();
    const mcts = new AdvancedMCTS(config);
    
    console.log('Starting position:');
    console.log(gameState.toString());
    console.log();
    
    let moveCount = 0;
    while (!gameState.isTerminal() && moveCount < 9) {
        const currentPlayer = gameState.getCurrentPlayer();
        console.log(`Player ${currentPlayer}'s turn`);
        
        // Use MCTS to find best move
        const bestAction = mcts.search(gameState, currentPlayer);
        
        if (bestAction === null) {
            console.log('No legal moves available');
            break;
        }
        
        // Apply the move
        gameState = gameState.applyAction(bestAction);
        moveCount++;
        
        console.log(`Player ${currentPlayer} plays [${bestAction[0]}, ${bestAction[1]}]`);
        console.log(gameState.toString());
        
        // Show search statistics
        const stats = mcts.getActionStatistics();
        console.log('MCTS Action Statistics:');
        for (const [action, stat] of Object.entries(stats)) {
            console.log(`  ${action}: visits=${stat.visits}, ` +
                       `avg_reward=${stat.averageReward.toFixed(3)}, ` +
                       `ucb1=${stat.ucb1Value.toFixed(3)}`);
        }
        console.log();
    }
    
    // Show final result
    if (gameState.isTerminal()) {
        const winner = gameState._checkWinner();
        if (winner === 1) {
            console.log('Player X (1) wins!');
        } else if (winner === -1) {
            console.log('Player O (-1) wins!');
        } else {
            console.log("It's a draw!");
        }
    }
    
    console.log(`Game completed in ${moveCount} moves`);
    console.log(`MCTS performed ${mcts.iterationCount} iterations per move on average`);
}

// Performance benchmark
function benchmarkMCTS() {
    console.log('\\n=== MCTS Performance Benchmark ===');
    
    const testCases = [
        { iterations: 1000, time: 1.0 },
        { iterations: 5000, time: 2.0 },
        { iterations: 10000, time: 5.0 }
    ];
    
    console.log('Test Case\\t\\tIterations\\tTime (s)\\tIterations/sec');
    console.log('-'.repeat(60));
    
    testCases.forEach((testCase, index) => {
        const config = {
            maxIterations: testCase.iterations,
            maxTimeSeconds: testCase.time
        };
        
        const gameState = new TicTacToeState();
        const mcts = new AdvancedMCTS(config);
        
        const startTime = Date.now();
        mcts.search(gameState, 1);
        const endTime = Date.now();
        
        const actualTime = (endTime - startTime) / 1000;
        const iterationsPerSecond = mcts.iterationCount / actualTime;
        
        console.log(`Case ${index + 1}\\t\\t\\t${mcts.iterationCount}\\t\\t${actualTime.toFixed(2)}\\t\\t${iterationsPerSecond.toFixed(0)}`);
    });
}

// Export for Node.js if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        GameState,
        TicTacToeState,
        MCTSNode,
        AdvancedMCTS,
        runTicTacToeExample,
        benchmarkMCTS
    };
}

// Run examples if this is the main script
if (typeof window === 'undefined' && typeof process !== 'undefined') {
    runTicTacToeExample();
    benchmarkMCTS();
}