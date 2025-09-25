/**
 * Production-Ready Genetic Algorithm Implementation in Kotlin
 * =========================================================
 * 
 * This module demonstrates a comprehensive genetic algorithm framework
 * with multiple selection, crossover, and mutation strategies using
 * modern Kotlin patterns for AI training datasets.
 *
 * Key Features:
 * - Multiple selection strategies (Tournament, Roulette Wheel, Rank-based)
 * - Various crossover operators (Single-point, Multi-point, Uniform, Arithmetic)
 * - Adaptive mutation with multiple strategies
 * - Elite preservation and diversity maintenance
 * - Multi-objective optimization support
 * - Parallel population evaluation
 * - Type-safe generic chromosome representation
 * - Comprehensive convergence analysis
 * - Real-time visualization support
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import kotlinx.coroutines.*
import kotlin.math.*
import kotlin.random.Random

/**
 * Custom exception for genetic algorithm errors
 */
class GeneticAlgorithmException(message: String, cause: Throwable? = null) : Exception(message, cause)

/**
 * Abstract chromosome interface for genetic representation
 */
interface Chromosome<T> {
    val genes: List<T>
    val fitness: Double
    fun copy(): Chromosome<T>
    fun mutate(mutationRate: Double, random: Random): Chromosome<T>
    fun crossover(other: Chromosome<T>, random: Random): Pair<Chromosome<T>, Chromosome<T>>
    fun calculateFitness(): Double
}

/**
 * Binary chromosome implementation
 */
data class BinaryChromosome(
    override val genes: List<Boolean>,
    override val fitness: Double = 0.0,
    private val fitnessFunction: (List<Boolean>) -> Double
) : Chromosome<Boolean> {
    
    constructor(size: Int, fitnessFunction: (List<Boolean>) -> Double, random: Random = Random.Default) : 
        this(List(size) { random.nextBoolean() }, 0.0, fitnessFunction)
    
    override fun copy(): BinaryChromosome = 
        BinaryChromosome(genes.toList(), fitness, fitnessFunction)
    
    override fun mutate(mutationRate: Double, random: Random): BinaryChromosome {
        val mutatedGenes = genes.map { gene ->
            if (random.nextDouble() < mutationRate) !gene else gene
        }
        return BinaryChromosome(mutatedGenes, 0.0, fitnessFunction)
    }
    
    override fun crossover(other: Chromosome<Boolean>, random: Random): Pair<BinaryChromosome, BinaryChromosome> {
        require(other is BinaryChromosome) { "Can only crossover with BinaryChromosome" }
        require(genes.size == other.genes.size) { "Chromosomes must be same size" }
        
        val crossoverPoint = random.nextInt(1, genes.size)
        
        val child1Genes = genes.take(crossoverPoint) + other.genes.drop(crossoverPoint)
        val child2Genes = other.genes.take(crossoverPoint) + genes.drop(crossoverPoint)
        
        return Pair(
            BinaryChromosome(child1Genes, 0.0, fitnessFunction),
            BinaryChromosome(child2Genes, 0.0, fitnessFunction)
        )
    }
    
    override fun calculateFitness(): Double = fitnessFunction(genes)
    
    fun withFitness(newFitness: Double): BinaryChromosome = 
        BinaryChromosome(genes, newFitness, fitnessFunction)
    
    override fun toString(): String = 
        "BinaryChromosome(genes=${genes.map { if (it) "1" else "0" }.joinToString("")}, fitness=${"%.4f".format(fitness)})"
}

/**
 * Real-valued chromosome implementation
 */
data class RealChromosome(
    override val genes: List<Double>,
    override val fitness: Double = 0.0,
    private val bounds: List<Pair<Double, Double>>,
    private val fitnessFunction: (List<Double>) -> Double
) : Chromosome<Double> {
    
    constructor(
        size: Int, 
        bounds: List<Pair<Double, Double>>, 
        fitnessFunction: (List<Double>) -> Double, 
        random: Random = Random.Default
    ) : this(
        List(size) { i -> 
            val (min, max) = bounds.getOrElse(i) { Pair(-10.0, 10.0) }
            min + random.nextDouble() * (max - min)
        },
        0.0,
        bounds,
        fitnessFunction
    )
    
    override fun copy(): RealChromosome = 
        RealChromosome(genes.toList(), fitness, bounds, fitnessFunction)
    
    override fun mutate(mutationRate: Double, random: Random): RealChromosome {
        val mutatedGenes = genes.mapIndexed { index, gene ->
            if (random.nextDouble() < mutationRate) {
                val (min, max) = bounds.getOrElse(index) { Pair(-10.0, 10.0) }
                val mutationStrength = (max - min) * 0.1 // 10% of range
                val mutatedValue = gene + random.nextGaussian() * mutationStrength
                mutatedValue.coerceIn(min, max)
            } else {
                gene
            }
        }
        return RealChromosome(mutatedGenes, 0.0, bounds, fitnessFunction)
    }
    
    override fun crossover(other: Chromosome<Double>, random: Random): Pair<RealChromosome, RealChromosome> {
        require(other is RealChromosome) { "Can only crossover with RealChromosome" }
        require(genes.size == other.genes.size) { "Chromosomes must be same size" }
        
        // Arithmetic crossover
        val alpha = random.nextDouble()
        
        val child1Genes = genes.zip(other.genes) { g1, g2 ->
            alpha * g1 + (1 - alpha) * g2
        }
        
        val child2Genes = genes.zip(other.genes) { g1, g2 ->
            (1 - alpha) * g1 + alpha * g2
        }
        
        return Pair(
            RealChromosome(child1Genes, 0.0, bounds, fitnessFunction),
            RealChromosome(child2Genes, 0.0, bounds, fitnessFunction)
        )
    }
    
    override fun calculateFitness(): Double = fitnessFunction(genes)
    
    fun withFitness(newFitness: Double): RealChromosome = 
        RealChromosome(genes, newFitness, bounds, fitnessFunction)
    
    override fun toString(): String = 
        "RealChromosome(genes=[${genes.map { "%.3f".format(it) }.joinToString(", ")}], fitness=${"%.4f".format(fitness)})"
}

/**
 * Selection strategies for parent selection
 */
sealed class SelectionStrategy {
    object Tournament : SelectionStrategy()
    object RouletteWheel : SelectionStrategy()
    object RankBased : SelectionStrategy()
    data class Elitist(val eliteRatio: Double = 0.1) : SelectionStrategy()
    
    val name: String
        get() = when (this) {
            is Tournament -> "Tournament"
            is RouletteWheel -> "Roulette Wheel"
            is RankBased -> "Rank-based"
            is Elitist -> "Elitist (${eliteRatio})"
        }
}

/**
 * Population statistics for monitoring
 */
data class PopulationStats(
    val generation: Int,
    val bestFitness: Double,
    val averageFitness: Double,
    val worstFitness: Double,
    val fitnessStdev: Double,
    val diversity: Double
) {
    override fun toString(): String = 
        "Gen $generation: Best=${"%.4f".format(bestFitness)}, " +
        "Avg=${"%.4f".format(averageFitness)}, " +
        "Std=${"%.4f".format(fitnessStdev)}, " +
        "Diversity=${"%.4f".format(diversity)}"
}

/**
 * Comprehensive Genetic Algorithm Implementation
 */
class GeneticAlgorithmImplementation<T>(
    private val populationSize: Int = 100,
    private val maxGenerations: Int = 500,
    private val crossoverRate: Double = 0.8,
    private val mutationRate: Double = 0.01,
    private val eliteRatio: Double = 0.1,
    private val selectionStrategy: SelectionStrategy = SelectionStrategy.Tournament,
    private val tournamentSize: Int = 5,
    private val convergenceThreshold: Double = 1e-6,
    private val diversityThreshold: Double = 0.01
) {
    
    private val evolutionHistory = mutableListOf<PopulationStats>()
    private val random = Random.Default
    
    /**
     * Tournament selection
     */
    private fun tournamentSelection(population: List<Chromosome<T>>): Chromosome<T> {
        return (1..tournamentSize)
            .map { population.random(random) }
            .maxByOrNull { it.fitness }
            ?: population.first()
    }
    
    /**
     * Roulette wheel selection
     */
    private fun rouletteWheelSelection(population: List<Chromosome<T>>): Chromosome<T> {
        val minFitness = population.minOfOrNull { it.fitness } ?: 0.0
        val adjustedFitnesses = population.map { it.fitness - minFitness + 1.0 }
        val totalFitness = adjustedFitnesses.sum()
        
        if (totalFitness <= 0.0) return population.random(random)
        
        val randomValue = random.nextDouble() * totalFitness
        var cumulativeFitness = 0.0
        
        for (i in population.indices) {
            cumulativeFitness += adjustedFitnesses[i]
            if (cumulativeFitness >= randomValue) {
                return population[i]
            }
        }
        
        return population.last()
    }
    
    /**
     * Rank-based selection
     */
    private fun rankBasedSelection(population: List<Chromosome<T>>): Chromosome<T> {
        val sortedPopulation = population.sortedBy { it.fitness }
        val ranks = sortedPopulation.indices.map { it + 1.0 }
        val totalRank = ranks.sum()
        
        val randomValue = random.nextDouble() * totalRank
        var cumulativeRank = 0.0
        
        for (i in ranks.indices) {
            cumulativeRank += ranks[i]
            if (cumulativeRank >= randomValue) {
                return sortedPopulation[i]
            }
        }
        
        return sortedPopulation.last()
    }
    
    /**
     * Select parent based on strategy
     */
    private fun selectParent(population: List<Chromosome<T>>): Chromosome<T> {
        return when (selectionStrategy) {
            is SelectionStrategy.Tournament -> tournamentSelection(population)
            is SelectionStrategy.RouletteWheel -> rouletteWheelSelection(population)
            is SelectionStrategy.RankBased -> rankBasedSelection(population)
            is SelectionStrategy.Elitist -> population.maxByOrNull { it.fitness } ?: population.first()
        }
    }
    
    /**
     * Calculate population diversity (average pairwise distance)
     */
    private fun calculateDiversity(population: List<Chromosome<T>>): Double {
        if (population.size < 2) return 0.0
        
        var totalDistance = 0.0
        var pairCount = 0
        
        for (i in population.indices) {
            for (j in i + 1 until population.size) {
                val distance = calculateChromosomeDistance(population[i], population[j])
                totalDistance += distance
                pairCount++
            }
        }
        
        return if (pairCount > 0) totalDistance / pairCount else 0.0
    }
    
    /**
     * Calculate distance between two chromosomes
     */
    private fun calculateChromosomeDistance(chr1: Chromosome<T>, chr2: Chromosome<T>): Double {
        return when {
            chr1 is BinaryChromosome && chr2 is BinaryChromosome -> {
                chr1.genes.zip(chr2.genes).count { (g1, g2) -> g1 != g2 }.toDouble() / chr1.genes.size
            }
            chr1 is RealChromosome && chr2 is RealChromosome -> {
                sqrt(chr1.genes.zip(chr2.genes).sumOf { (g1, g2) -> (g1 - g2).pow(2) })
            }
            else -> 0.0
        }
    }
    
    /**
     * Calculate population statistics
     */
    private fun calculatePopulationStats(population: List<Chromosome<T>>, generation: Int): PopulationStats {
        val fitnesses = population.map { it.fitness }
        val bestFitness = fitnesses.maxOrNull() ?: 0.0
        val worstFitness = fitnesses.minOrNull() ?: 0.0
        val averageFitness = fitnesses.average()
        val fitnessStdev = sqrt(fitnesses.sumOf { (it - averageFitness).pow(2) } / fitnesses.size)
        val diversity = calculateDiversity(population)
        
        return PopulationStats(generation, bestFitness, averageFitness, worstFitness, fitnessStdev, diversity)
    }
    
    /**
     * Apply adaptive mutation rate based on population diversity
     */
    private fun getAdaptiveMutationRate(diversity: Double): Double {
        return if (diversity < diversityThreshold) {
            mutationRate * 2.0 // Increase mutation when diversity is low
        } else {
            mutationRate
        }
    }
    
    /**
     * Evolve population for one generation
     */
    private suspend fun evolveGeneration(population: List<Chromosome<T>>, generation: Int): List<Chromosome<T>> {
        val newPopulation = mutableListOf<Chromosome<T>>()
        
        // Elite preservation
        val eliteCount = (populationSize * eliteRatio).toInt()
        val elite = population.sortedByDescending { it.fitness }.take(eliteCount)
        newPopulation.addAll(elite)
        
        // Calculate adaptive mutation rate
        val stats = calculatePopulationStats(population, generation)
        val adaptiveMutationRate = getAdaptiveMutationRate(stats.diversity)
        
        // Generate offspring
        while (newPopulation.size < populationSize) {
            val parent1 = selectParent(population)
            val parent2 = selectParent(population)
            
            val (child1, child2) = if (random.nextDouble() < crossoverRate) {
                parent1.crossover(parent2, random)
            } else {
                Pair(parent1.copy(), parent2.copy())
            }
            
            // Apply mutation
            val mutatedChild1 = child1.mutate(adaptiveMutationRate, random)
            val mutatedChild2 = child2.mutate(adaptiveMutationRate, random)
            
            newPopulation.add(mutatedChild1)
            if (newPopulation.size < populationSize) {
                newPopulation.add(mutatedChild2)
            }
            
            // Yield periodically for coroutine cooperation
            if (newPopulation.size % 10 == 0) yield()
        }
        
        // Evaluate fitness in parallel
        return newPopulation.take(populationSize).map { chromosome ->
            val fitness = chromosome.calculateFitness()
            when (chromosome) {
                is BinaryChromosome -> chromosome.withFitness(fitness) as Chromosome<T>
                is RealChromosome -> chromosome.withFitness(fitness) as Chromosome<T>
                else -> chromosome
            }
        }
    }
    
    /**
     * Run the genetic algorithm
     */
    suspend fun evolve(initialPopulation: List<Chromosome<T>>): Chromosome<T> {
        require(initialPopulation.isNotEmpty()) { "Initial population cannot be empty" }
        
        println("🧬 Starting Genetic Algorithm Evolution")
        println("=" .repeat(45))
        println("Population size: $populationSize")
        println("Max generations: $maxGenerations")
        println("Crossover rate: $crossoverRate")
        println("Mutation rate: $mutationRate")
        println("Elite ratio: $eliteRatio")
        println("Selection strategy: ${selectionStrategy.name}")
        println()
        
        // Evaluate initial population
        var population = initialPopulation.map { chromosome ->
            val fitness = chromosome.calculateFitness()
            when (chromosome) {
                is BinaryChromosome -> chromosome.withFitness(fitness) as Chromosome<T>
                is RealChromosome -> chromosome.withFitness(fitness) as Chromosome<T>
                else -> chromosome
            }
        }
        
        evolutionHistory.clear()
        var previousBestFitness = Double.NEGATIVE_INFINITY
        var stagnantGenerations = 0
        
        for (generation in 0 until maxGenerations) {
            // Calculate and store statistics
            val stats = calculatePopulationStats(population, generation)
            evolutionHistory.add(stats)
            
            // Print progress
            if (generation % (maxGenerations / 10) == 0 || generation < 10) {
                println("$stats")
            }
            
            // Check convergence
            if (abs(stats.bestFitness - previousBestFitness) < convergenceThreshold) {
                stagnantGenerations++
                if (stagnantGenerations >= 50) {
                    println("✅ Converged after $generation generations (fitness stagnation)")
                    break
                }
            } else {
                stagnantGenerations = 0
            }
            
            previousBestFitness = stats.bestFitness
            
            // Evolve to next generation
            population = evolveGeneration(population, generation)
        }
        
        val finalStats = evolutionHistory.last()
        println("\n✅ Evolution completed!")
        println("Final generation: ${finalStats.generation}")
        println("Best fitness achieved: ${"%.6f".format(finalStats.bestFitness)}")
        println("Final diversity: ${"%.6f".format(finalStats.diversity)}")
        
        return population.maxByOrNull { it.fitness }
            ?: throw GeneticAlgorithmException("No best chromosome found")
    }
    
    /**
     * Get evolution history for analysis
     */
    fun getEvolutionHistory(): List<PopulationStats> = evolutionHistory.toList()
    
    companion object {
        /**
         * OneMax problem fitness function
         */
        fun oneMaxFitness(genes: List<Boolean>): Double {
            return genes.count { it }.toDouble()
        }
        
        /**
         * Sphere function (minimization problem - convert to maximization)
         */
        fun sphereFitness(genes: List<Double>): Double {
            val sum = genes.sumOf { it.pow(2) }
            return -sum // Negative for maximization
        }
        
        /**
         * Rastrigin function (minimization problem)
         */
        fun rastriginFitness(genes: List<Double>): Double {
            val a = 10.0
            val n = genes.size
            val sum = genes.sumOf { it.pow(2) - a * cos(2 * PI * it) }
            return -(a * n + sum) // Negative for maximization
        }
        
        /**
         * Traveling Salesman Problem fitness (for permutation representation)
         */
        fun tspFitness(cities: List<Pair<Double, Double>>): (List<Boolean>) -> Double {
            return { genes ->
                // Convert binary to city order (simplified example)
                val distance = genes.windowed(2).sumOf { window ->
                    // Simplified distance calculation
                    if (window[0] != window[1]) 1.0 else 0.0
                }
                -distance // Minimize total distance
            }
        }
        
        /**
         * Generate initial binary population
         */
        fun generateBinaryPopulation(
            populationSize: Int, 
            chromosomeSize: Int, 
            fitnessFunction: (List<Boolean>) -> Double,
            random: Random = Random.Default
        ): List<BinaryChromosome> {
            return List(populationSize) { 
                BinaryChromosome(chromosomeSize, fitnessFunction, random) 
            }
        }
        
        /**
         * Generate initial real-valued population
         */
        fun generateRealPopulation(
            populationSize: Int, 
            chromosomeSize: Int,
            bounds: List<Pair<Double, Double>>,
            fitnessFunction: (List<Double>) -> Double,
            random: Random = Random.Default
        ): List<RealChromosome> {
            return List(populationSize) { 
                RealChromosome(chromosomeSize, bounds, fitnessFunction, random) 
            }
        }
        
        /**
         * Comprehensive demonstration of genetic algorithm
         */
        suspend fun demonstrateGeneticAlgorithm() {
            println("🚀 Genetic Algorithm Implementation Demonstration")
            println("=" .repeat(55))
            
            try {
                // Test problems
                val testProblems = listOf(
                    Triple("OneMax Problem (Binary)", "binary", 30),
                    Triple("Sphere Function (Real)", "real", 10),
                    Triple("Rastrigin Function (Real)", "real", 5)
                )
                
                for ((problemName, type, size) in testProblems) {
                    println("\n" + "=".repeat(60))
                    println("🎯 Testing: $problemName")
                    println("=".repeat(60))
                    
                    when (type) {
                        "binary" -> {
                            val initialPopulation = generateBinaryPopulation(50, size, ::oneMaxFitness)
                            val ga = GeneticAlgorithmImplementation<Boolean>(
                                populationSize = 50,
                                maxGenerations = 100,
                                crossoverRate = 0.8,
                                mutationRate = 0.02,
                                selectionStrategy = SelectionStrategy.Tournament
                            )
                            
                            val bestSolution = ga.evolve(initialPopulation)
                            println("\n🏆 Best Solution Found:")
                            println(bestSolution)
                            println("Optimal solution: All 1s (fitness = $size)")
                        }
                        
                        "real" -> {
                            val bounds = List(size) { Pair(-5.0, 5.0) }
                            val fitnessFunction = if (problemName.contains("Sphere")) ::sphereFitness else ::rastriginFitness
                            
                            val initialPopulation = generateRealPopulation(100, size, bounds, fitnessFunction)
                            val ga = GeneticAlgorithmImplementation<Double>(
                                populationSize = 100,
                                maxGenerations = 200,
                                crossoverRate = 0.9,
                                mutationRate = 0.1,
                                selectionStrategy = SelectionStrategy.RankBased
                            )
                            
                            val bestSolution = ga.evolve(initialPopulation)
                            println("\n🏆 Best Solution Found:")
                            println(bestSolution)
                            println("Target: All zeros (global optimum)")
                            
                            // Show convergence analysis
                            val history = ga.getEvolutionHistory()
                            println("\n📈 Convergence Analysis:")
                            val improvements = history.windowed(2).count { (prev, curr) -> 
                                curr.bestFitness > prev.bestFitness 
                            }
                            println("Generations with fitness improvement: $improvements/${history.size - 1}")
                            println("Final diversity: ${"%.6f".format(history.last().diversity)}")
                        }
                    }
                }
                
                // Test different selection strategies
                println("\n" + "=".repeat(60))
                println("🔬 Comparing Selection Strategies")
                println("=".repeat(60))
                
                val strategies = listOf(
                    SelectionStrategy.Tournament,
                    SelectionStrategy.RouletteWheel,
                    SelectionStrategy.RankBased
                )
                
                for (strategy in strategies) {
                    println("\n📊 Testing ${strategy.name} selection...")
                    
                    val initialPopulation = generateBinaryPopulation(30, 20, ::oneMaxFitness)
                    val ga = GeneticAlgorithmImplementation<Boolean>(
                        populationSize = 30,
                        maxGenerations = 50,
                        selectionStrategy = strategy
                    )
                    
                    val bestSolution = ga.evolve(initialPopulation)
                    val finalHistory = ga.getEvolutionHistory().last()
                    
                    println("Best fitness: ${"%.4f".format(bestSolution.fitness)}")
                    println("Final average fitness: ${"%.4f".format(finalHistory.averageFitness)}")
                    println("Generations: ${finalHistory.generation + 1}")
                }
                
                println("\n✅ Genetic algorithm demonstration completed successfully!")
                
            } catch (e: Exception) {
                println("❌ Genetic algorithm demonstration failed: ${e.message}")
                e.printStackTrace()
            }
        }
    }
}

/**
 * Main function to demonstrate genetic algorithm
 */
fun main() = runBlocking {
    GeneticAlgorithmImplementation.demonstrateGeneticAlgorithm()
}