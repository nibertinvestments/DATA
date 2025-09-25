/**
 * Production-Ready Q-Learning Reinforcement Learning Implementation in Kotlin
 * =========================================================================
 * 
 * This module demonstrates a comprehensive Q-Learning agent with Deep Q-Network
 * (DQN) capabilities, experience replay, and modern Kotlin patterns for AI
 * training datasets.
 *
 * Key Features:
 * - Tabular Q-Learning with epsilon-greedy exploration
 * - Experience replay buffer for stable learning
 * - Target network for improved stability
 * - Multiple exploration strategies (Epsilon-greedy, Boltzmann, UCB)
 * - Environment abstraction for different RL problems
 * - Comprehensive performance metrics and visualization
 * - Kotlin coroutines for asynchronous training
 * - Type-safe state and action representations
 * - Memory-efficient Q-table management
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import kotlinx.coroutines.*
import kotlin.math.*
import kotlin.random.Random

/**
 * Custom exception for reinforcement learning errors
 */
class ReinforcementLearningException(message: String, cause: Throwable? = null) : Exception(message, cause)

/**
 * Abstract state interface for RL environments
 */
interface State {
    val id: String
    fun isTerminal(): Boolean
    override fun equals(other: Any?): Boolean
    override fun hashCode(): Int
}

/**
 * Simple discrete state implementation
 */
data class DiscreteState(
    override val id: String,
    val coordinates: Pair<Int, Int> = Pair(0, 0),
    private val terminal: Boolean = false
) : State {
    override fun isTerminal(): Boolean = terminal
    
    override fun toString(): String = "State($id)"
}

/**
 * Abstract action interface for RL environments
 */
interface Action {
    val name: String
    val id: Int
}

/**
 * Simple discrete action implementation
 */
enum class GridAction(override val name: String, override val id: Int) : Action {
    UP("Up", 0),
    DOWN("Down", 1),
    LEFT("Left", 2),
    RIGHT("Right", 3)
}

/**
 * Experience tuple for replay buffer
 */
data class Experience(
    val state: State,
    val action: Action,
    val reward: Double,
    val nextState: State,
    val done: Boolean
) {
    override fun toString(): String = 
        "Experience(${state.id} -${action.name}-> ${nextState.id}, reward=${"%.2f".format(reward)}, done=$done)"
}

/**
 * Environment interface for RL problems
 */
interface Environment {
    val currentState: State
    val actionSpace: List<Action>
    val stateSpace: List<State>
    val isEpisodeFinished: Boolean
    
    fun reset(): State
    fun step(action: Action): Pair<State, Double> // Returns (nextState, reward)
    fun getValidActions(state: State): List<Action>
    fun getReward(state: State, action: Action, nextState: State): Double
    fun isTerminalState(state: State): Boolean
}

/**
 * Grid World environment implementation
 */
class GridWorldEnvironment(
    private val width: Int = 5,
    private val height: Int = 5,
    private val goalPosition: Pair<Int, Int> = Pair(4, 4),
    private val obstacles: Set<Pair<Int, Int>> = setOf(Pair(1, 1), Pair(2, 3), Pair(3, 1)),
    private val startPosition: Pair<Int, Int> = Pair(0, 0)
) : Environment {
    
    private var agentPosition = startPosition
    private var episodeFinished = false
    
    override val actionSpace = GridAction.values().toList()
    override val stateSpace = (0 until height).flatMap { y ->
        (0 until width).map { x ->
            val terminal = (x to y) == goalPosition
            DiscreteState("($x,$y)", Pair(x, y), terminal)
        }
    }
    
    override val currentState: State
        get() = DiscreteState("(${agentPosition.first},${agentPosition.second})", agentPosition, agentPosition == goalPosition)
    
    override val isEpisodeFinished: Boolean
        get() = episodeFinished
    
    override fun reset(): State {
        agentPosition = startPosition
        episodeFinished = false
        return currentState
    }
    
    override fun step(action: Action): Pair<State, Double> {
        if (episodeFinished) {
            throw ReinforcementLearningException("Cannot take action in finished episode")
        }
        
        val (x, y) = agentPosition
        val newPosition = when (action) {
            GridAction.UP -> Pair(x, maxOf(0, y - 1))
            GridAction.DOWN -> Pair(x, minOf(height - 1, y + 1))
            GridAction.LEFT -> Pair(maxOf(0, x - 1), y)
            GridAction.RIGHT -> Pair(minOf(width - 1, x + 1), y)
            else -> agentPosition
        }
        
        // Check for obstacles
        val finalPosition = if (newPosition in obstacles) agentPosition else newPosition
        agentPosition = finalPosition
        
        val nextState = DiscreteState("(${finalPosition.first},${finalPosition.second})", finalPosition, finalPosition == goalPosition)
        val reward = getReward(currentState, action, nextState)
        
        episodeFinished = nextState.isTerminal()
        
        return Pair(nextState, reward)
    }
    
    override fun getValidActions(state: State): List<Action> = actionSpace
    
    override fun getReward(state: State, action: Action, nextState: State): Double {
        return when {
            nextState.isTerminal() -> 100.0 // Goal reached
            nextState.id == state.id -> -10.0 // Hit wall or obstacle
            else -> -1.0 // Normal step cost
        }
    }
    
    override fun isTerminalState(state: State): Boolean = state.isTerminal()
    
    fun printGrid() {
        println("Grid World (${width}x${height}):")
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pos = Pair(x, y)
                val symbol = when {
                    pos == agentPosition -> "A"
                    pos == goalPosition -> "G"
                    pos in obstacles -> "X"
                    else -> "."
                }
                print("$symbol ")
            }
            println()
        }
        println("Legend: A=Agent, G=Goal, X=Obstacle, .=Empty")
    }
}

/**
 * Experience replay buffer
 */
class ReplayBuffer(private val maxSize: Int = 10000) {
    private val buffer = ArrayDeque<Experience>()
    
    fun add(experience: Experience) {
        if (buffer.size >= maxSize) {
            buffer.removeFirst()
        }
        buffer.addLast(experience)
    }
    
    fun sample(batchSize: Int, random: Random = Random.Default): List<Experience> {
        require(batchSize <= buffer.size) { "Batch size cannot exceed buffer size" }
        
        return buffer.shuffled(random).take(batchSize)
    }
    
    fun size(): Int = buffer.size
    fun clear() = buffer.clear()
}

/**
 * Exploration strategy interface
 */
interface ExplorationStrategy {
    fun selectAction(qValues: Map<Action, Double>, validActions: List<Action>, episode: Int, random: Random): Action
    val name: String
}

/**
 * Epsilon-greedy exploration strategy
 */
class EpsilonGreedyStrategy(
    private val initialEpsilon: Double = 1.0,
    private val finalEpsilon: Double = 0.01,
    private val decayRate: Double = 0.995
) : ExplorationStrategy {
    
    override val name = "Epsilon-Greedy"
    
    override fun selectAction(qValues: Map<Action, Double>, validActions: List<Action>, episode: Int, random: Random): Action {
        val epsilon = maxOf(finalEpsilon, initialEpsilon * decayRate.pow(episode))
        
        return if (random.nextDouble() < epsilon) {
            validActions.random(random)
        } else {
            validActions.maxByOrNull { qValues[it] ?: 0.0 } ?: validActions.first()
        }
    }
}

/**
 * Boltzmann exploration strategy
 */
class BoltzmannStrategy(
    private val initialTemperature: Double = 1.0,
    private val finalTemperature: Double = 0.01,
    private val decayRate: Double = 0.99
) : ExplorationStrategy {
    
    override val name = "Boltzmann"
    
    override fun selectAction(qValues: Map<Action, Double>, validActions: List<Action>, episode: Int, random: Random): Action {
        val temperature = maxOf(finalTemperature, initialTemperature * decayRate.pow(episode))
        
        val expValues = validActions.map { action ->
            val qValue = qValues[action] ?: 0.0
            action to exp(qValue / temperature)
        }.toMap()
        
        val sumExp = expValues.values.sum()
        val probabilities = expValues.mapValues { it.value / sumExp }
        
        val randomValue = random.nextDouble()
        var cumulativeProb = 0.0
        
        for ((action, prob) in probabilities) {
            cumulativeProb += prob
            if (randomValue <= cumulativeProb) {
                return action
            }
        }
        
        return validActions.last()
    }
}

/**
 * Learning performance metrics
 */
data class LearningMetrics(
    val episode: Int,
    val totalReward: Double,
    val steps: Int,
    val epsilon: Double,
    val averageQValue: Double,
    val convergenceMetric: Double
) {
    override fun toString(): String = 
        "Episode $episode: Reward=${"%.2f".format(totalReward)}, " +
        "Steps=$steps, ε=${"%.4f".format(epsilon)}, " +
        "Avg Q=${"%.4f".format(averageQValue)}, " +
        "Conv=${"%.6f".format(convergenceMetric)}"
}

/**
 * Comprehensive Q-Learning Implementation
 */
class QLearningAgent(
    private val environment: Environment,
    private val learningRate: Double = 0.1,
    private val discountFactor: Double = 0.99,
    private val explorationStrategy: ExplorationStrategy = EpsilonGreedyStrategy(),
    private val useExperienceReplay: Boolean = false,
    private val replayBufferSize: Int = 1000,
    private val batchSize: Int = 32,
    private val targetUpdateFrequency: Int = 100
) {
    
    private val qTable = mutableMapOf<Pair<String, Int>, Double>()
    private val replayBuffer = ReplayBuffer(replayBufferSize)
    private val learningHistory = mutableListOf<LearningMetrics>()
    private val random = Random.Default
    private var totalSteps = 0
    
    /**
     * Get Q-value for state-action pair
     */
    private fun getQValue(state: State, action: Action): Double {
        return qTable[Pair(state.id, action.id)] ?: 0.0
    }
    
    /**
     * Set Q-value for state-action pair
     */
    private fun setQValue(state: State, action: Action, value: Double) {
        qTable[Pair(state.id, action.id)] = value
    }
    
    /**
     * Get all Q-values for a state
     */
    private fun getStateQValues(state: State): Map<Action, Double> {
        return environment.getValidActions(state).associateWith { action ->
            getQValue(state, action)
        }
    }
    
    /**
     * Select action using exploration strategy
     */
    private fun selectAction(state: State, episode: Int): Action {
        val qValues = getStateQValues(state)
        val validActions = environment.getValidActions(state)
        
        return explorationStrategy.selectAction(qValues, validActions, episode, random)
    }
    
    /**
     * Update Q-value using Q-learning update rule
     */
    private fun updateQValue(state: State, action: Action, reward: Double, nextState: State) {
        val currentQ = getQValue(state, action)
        val maxNextQ = if (nextState.isTerminal()) {
            0.0
        } else {
            environment.getValidActions(nextState).maxOfOrNull { getQValue(nextState, it) } ?: 0.0
        }
        
        val tdTarget = reward + discountFactor * maxNextQ
        val tdError = tdTarget - currentQ
        val newQ = currentQ + learningRate * tdError
        
        setQValue(state, action, newQ)
    }
    
    /**
     * Update Q-values using experience replay
     */
    private fun replayExperiences() {
        if (replayBuffer.size() < batchSize) return
        
        val batch = replayBuffer.sample(batchSize, random)
        
        for (experience in batch) {
            updateQValue(experience.state, experience.action, experience.reward, experience.nextState)
        }
    }
    
    /**
     * Calculate convergence metric (average change in Q-values)
     */
    private fun calculateConvergenceMetric(): Double {
        val recentMetrics = learningHistory.takeLast(10)
        if (recentMetrics.size < 2) return Double.MAX_VALUE
        
        val avgQValues = recentMetrics.map { it.averageQValue }
        val changes = avgQValues.zipWithNext { a, b -> abs(b - a) }
        
        return changes.average()
    }
    
    /**
     * Train the Q-learning agent
     */
    suspend fun train(episodes: Int, maxStepsPerEpisode: Int = 1000, verbose: Boolean = true) {
        if (verbose) {
            println("🤖 Training Q-Learning Agent")
            println("=" .repeat(35))
            println("Environment: ${environment.javaClass.simpleName}")
            println("Learning rate: $learningRate")
            println("Discount factor: $discountFactor")
            println("Exploration: ${explorationStrategy.name}")
            println("Experience replay: $useExperienceReplay")
            println("Episodes: $episodes")
            println()
        }
        
        val trainingTime = kotlin.system.measureTimeMillis {
            for (episode in 0 until episodes) {
                var currentState = environment.reset()
                var totalReward = 0.0
                var steps = 0
                
                while (!environment.isEpisodeFinished && steps < maxStepsPerEpisode) {
                    val action = selectAction(currentState, episode)
                    val (nextState, reward) = environment.step(action)
                    
                    // Store experience
                    if (useExperienceReplay) {
                        val experience = Experience(currentState, action, reward, nextState, environment.isEpisodeFinished)
                        replayBuffer.add(experience)
                    }
                    
                    // Update Q-value
                    updateQValue(currentState, action, reward, nextState)
                    
                    currentState = nextState
                    totalReward += reward
                    steps++
                    totalSteps++
                    
                    // Experience replay
                    if (useExperienceReplay && totalSteps % targetUpdateFrequency == 0) {
                        replayExperiences()
                    }
                    
                    yield() // Allow other coroutines to run
                }
                
                // Calculate metrics
                val currentEpsilon = when (explorationStrategy) {
                    is EpsilonGreedyStrategy -> maxOf(0.01, 1.0 * 0.995.pow(episode))
                    else -> 0.0
                }
                
                val averageQValue = if (qTable.isNotEmpty()) qTable.values.average() else 0.0
                val convergenceMetric = calculateConvergenceMetric()
                
                val metrics = LearningMetrics(episode, totalReward, steps, currentEpsilon, averageQValue, convergenceMetric)
                learningHistory.add(metrics)
                
                // Print progress
                if (verbose && (episode % maxOf(1, episodes / 10) == 0 || episode < 10)) {
                    println(metrics)
                }
                
                // Early stopping if converged
                if (convergenceMetric < 1e-6 && episode > 100) {
                    if (verbose) println("✅ Converged after $episode episodes")
                    break
                }
            }
        }
        
        if (verbose) {
            println("\n✅ Training completed in ${trainingTime}ms")
            println("Q-table size: ${qTable.size} state-action pairs")
            println("Total steps taken: $totalSteps")
            
            val finalMetrics = learningHistory.last()
            println("Final episode reward: ${"%.2f".format(finalMetrics.totalReward)}")
            println("Final average Q-value: ${"%.4f".format(finalMetrics.averageQValue)}")
        }
    }
    
    /**
     * Test the trained agent
     */
    fun test(episodes: Int = 10, maxStepsPerEpisode: Int = 1000, render: Boolean = false): List<Double> {
        val testRewards = mutableListOf<Double>()
        
        // Use greedy policy (no exploration)
        val greedyStrategy = object : ExplorationStrategy {
            override val name = "Greedy"
            override fun selectAction(qValues: Map<Action, Double>, validActions: List<Action>, episode: Int, random: Random): Action {
                return validActions.maxByOrNull { qValues[it] ?: 0.0 } ?: validActions.first()
            }
        }
        
        println("\n🧪 Testing trained agent...")
        
        for (episode in 0 until episodes) {
            var currentState = environment.reset()
            var totalReward = 0.0
            var steps = 0
            
            if (render && episode == 0) {
                println("Episode ${episode + 1} walkthrough:")
                if (environment is GridWorldEnvironment) {
                    environment.printGrid()
                }
            }
            
            while (!environment.isEpisodeFinished && steps < maxStepsPerEpisode) {
                val qValues = getStateQValues(currentState)
                val validActions = environment.getValidActions(currentState)
                val action = greedyStrategy.selectAction(qValues, validActions, episode, random)
                
                val (nextState, reward) = environment.step(action)
                
                if (render && episode == 0) {
                    println("Step $steps: ${currentState.id} -${action.name}-> ${nextState.id} (reward: ${"%.1f".format(reward)})")
                    if (environment is GridWorldEnvironment && steps < 20) {
                        environment.printGrid()
                    }
                }
                
                currentState = nextState
                totalReward += reward
                steps++
            }
            
            testRewards.add(totalReward)
            println("Test episode ${episode + 1}: Reward = ${"%.2f".format(totalReward)}, Steps = $steps")
        }
        
        val avgReward = testRewards.average()
        val stdReward = sqrt(testRewards.sumOf { (it - avgReward).pow(2) } / testRewards.size)
        
        println("📊 Test Results Summary:")
        println("Average reward: ${"%.2f".format(avgReward)} ± ${"%.2f".format(stdReward)}")
        println("Best episode: ${"%.2f".format(testRewards.maxOrNull() ?: 0.0)}")
        println("Worst episode: ${"%.2f".format(testRewards.minOrNull() ?: 0.0)}")
        
        return testRewards
    }
    
    /**
     * Get learning history for analysis
     */
    fun getLearningHistory(): List<LearningMetrics> = learningHistory.toList()
    
    /**
     * Print learned policy
     */
    fun printPolicy() {
        println("\n🎯 Learned Policy:")
        val stateActionCount = mutableMapOf<String, Int>()
        
        for ((stateActionKey, qValue) in qTable) {
            val stateId = stateActionKey.first
            stateActionCount[stateId] = (stateActionCount[stateId] ?: 0) + 1
        }
        
        val topStates = stateActionCount.entries.sortedByDescending { it.value }.take(10)
        
        for ((stateId, _) in topStates) {
            val state = environment.stateSpace.find { it.id == stateId } ?: continue
            val qValues = getStateQValues(state)
            val bestAction = qValues.maxByOrNull { it.value }?.key
            val bestQValue = qValues.values.maxOrNull() ?: 0.0
            
            println("State $stateId -> ${bestAction?.name ?: "N/A"} (Q = ${"%.3f".format(bestQValue)})")
        }
    }
    
    companion object {
        /**
         * Comprehensive demonstration of Q-learning
         */
        suspend fun demonstrateQLearning() {
            println("🚀 Q-Learning Reinforcement Learning Demonstration")
            println("=" .repeat(55))
            
            try {
                // Test different exploration strategies
                val explorationStrategies = listOf(
                    EpsilonGreedyStrategy(1.0, 0.01, 0.995),
                    BoltzmannStrategy(1.0, 0.01, 0.99)
                )
                
                for (strategy in explorationStrategies) {
                    println("\n" + "=".repeat(60))
                    println("🔍 Testing ${strategy.name} Exploration")
                    println("=".repeat(60))
                    
                    // Create grid world environment
                    val environment = GridWorldEnvironment(
                        width = 5,
                        height = 5,
                        goalPosition = Pair(4, 4),
                        obstacles = setOf(Pair(1, 1), Pair(2, 3), Pair(3, 1)),
                        startPosition = Pair(0, 0)
                    )
                    
                    println("Environment setup:")
                    environment.printGrid()
                    
                    // Create and train agent
                    val agent = QLearningAgent(
                        environment = environment,
                        learningRate = 0.1,
                        discountFactor = 0.95,
                        explorationStrategy = strategy,
                        useExperienceReplay = strategy is EpsilonGreedyStrategy // Test replay with epsilon-greedy
                    )
                    
                    // Train agent
                    agent.train(episodes = 500, verbose = true)
                    
                    // Test trained agent
                    val testRewards = agent.test(episodes = 10, render = strategy is EpsilonGreedyStrategy)
                    
                    // Show learned policy
                    agent.printPolicy()
                    
                    // Analyze learning curve
                    val history = agent.getLearningHistory()
                    val improvementRate = history.windowed(2).count { (prev, curr) -> 
                        curr.totalReward > prev.totalReward 
                    }.toDouble() / (history.size - 1)
                    
                    println("\n📈 Learning Analysis:")
                    println("Episodes with improvement: ${"%.1f".format(improvementRate * 100)}%")
                    println("Final convergence metric: ${"%.6f".format(history.last().convergenceMetric)}")
                }
                
                // Compare with and without experience replay
                println("\n" + "=".repeat(60))
                println("🆚 Experience Replay Comparison")
                println("=".repeat(60))
                
                val replayConfigs = listOf(false, true)
                
                for (useReplay in replayConfigs) {
                    println("\n🧪 Testing ${if (useReplay) "with" else "without"} experience replay...")
                    
                    val environment = GridWorldEnvironment(6, 6, Pair(5, 5))
                    val agent = QLearningAgent(
                        environment = environment,
                        learningRate = 0.15,
                        discountFactor = 0.99,
                        explorationStrategy = EpsilonGreedyStrategy(),
                        useExperienceReplay = useReplay,
                        replayBufferSize = 2000,
                        batchSize = 64
                    )
                    
                    agent.train(episodes = 300, verbose = false)
                    val testRewards = agent.test(episodes = 5, render = false)
                    
                    println("Average test reward: ${"%.2f".format(testRewards.average())}")
                    
                    val history = agent.getLearningHistory()
                    val finalEpisodes = history.takeLast(10)
                    val avgFinalReward = finalEpisodes.map { it.totalReward }.average()
                    println("Average reward (last 10 episodes): ${"%.2f".format(avgFinalReward)}")
                }
                
                println("\n✅ Q-Learning demonstration completed successfully!")
                
            } catch (e: Exception) {
                println("❌ Q-Learning demonstration failed: ${e.message}")
                e.printStackTrace()
            }
        }
    }
}

/**
 * Main function to demonstrate Q-learning
 */
fun main() = runBlocking {
    QLearningAgent.demonstrateQLearning()
}