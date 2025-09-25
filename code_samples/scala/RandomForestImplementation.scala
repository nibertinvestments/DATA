/**
 * Random Forest Implementation in Scala
 * =====================================
 * 
 * This module demonstrates production-ready Random Forest implementation in Scala
 * with functional programming patterns, immutable data structures, and comprehensive
 * error handling for AI training datasets.
 *
 * Key Features:
 * - Functional programming with immutable data structures
 * - Akka actors for parallel tree training
 * - Case classes and pattern matching for type safety
 * - ScalaTest integration for comprehensive testing
 * - Future-based asynchronous operations
 * - Comprehensive error handling with Try/Either monads
 * - Extensive documentation for AI learning
 * - Production-ready patterns with Scala ecosystem integration
 * 
 * Author: AI Training Dataset
 * License: MIT
 */

import scala.util.{Try, Success, Failure, Random}
import scala.concurrent.{Future, ExecutionContext, Await}
import scala.concurrent.duration._
import scala.collection.{mutable, immutable}
import scala.math.{log, sqrt, pow}
import akka.actor.{ActorSystem, Actor, Props, ActorRef}
import akka.pattern.ask
import akka.util.Timeout

object RandomForestML {
  
  // Custom exceptions for the Random Forest
  sealed trait MLError extends Exception {
    def message: String
    override def getMessage: String = message
  }
  
  case class ValidationError(message: String) extends MLError
  case class TrainingError(message: String) extends MLError  
  case class PredictionError(message: String) extends MLError
  
  // Data structures using case classes for immutability
  case class DataPoint(features: Vector[Double], label: Double) {
    require(features.nonEmpty, "Features cannot be empty")
    require(!features.exists(_.isNaN), "Features cannot contain NaN values")
    require(!label.isNaN, "Label cannot be NaN")
  }
  
  case class Dataset(points: Vector[DataPoint]) {
    require(points.nonEmpty, "Dataset cannot be empty")
    
    lazy val featureCount: Int = points.head.features.length
    lazy val size: Int = points.length
    
    def splitByFeature(featureIndex: Int, threshold: Double): (Dataset, Dataset) = {
      val (left, right) = points.partition(_.features(featureIndex) <= threshold)
      (Dataset(left), Dataset(right))
    }
    
    def bootstrap(sampleSize: Int = points.length)(implicit random: Random): Dataset = {
      val bootstrapped = (1 to sampleSize).map(_ => points(random.nextInt(points.length)))
      Dataset(bootstrapped.toVector)
    }
    
    def randomFeatureSubset(count: Int)(implicit random: Random): Vector[Int] = {
      random.shuffle((0 until featureCount).toVector).take(count)
    }
  }
  
  // Decision Tree Node using sealed trait for pattern matching
  sealed trait TreeNode {
    def predict(features: Vector[Double]): Double
    def depth: Int
    def leafCount: Int
  }
  
  case class InternalNode(
    featureIndex: Int,
    threshold: Double,
    left: TreeNode,
    right: TreeNode
  ) extends TreeNode {
    
    def predict(features: Vector[Double]): Double = {
      if (features(featureIndex) <= threshold) left.predict(features)
      else right.predict(features)
    }
    
    lazy val depth: Int = 1 + math.max(left.depth, right.depth)
    lazy val leafCount: Int = left.leafCount + right.leafCount
  }
  
  case class LeafNode(prediction: Double) extends TreeNode {
    def predict(features: Vector[Double]): Double = prediction
    val depth: Int = 0
    val leafCount: Int = 1
  }
  
  // Tree metrics case class
  case class TreeMetrics(
    depth: Int,
    leafCount: Int,
    trainingAccuracy: Double,
    oobError: Option[Double] = None
  )
  
  // Decision Tree implementation with functional patterns
  class DecisionTree(
    maxDepth: Int = 10,
    minSamplesLeaf: Int = 1,
    minSamplesSplit: Int = 2
  )(implicit random: Random) {
    
    def train(dataset: Dataset): Try[TreeNode] = Try {
      require(dataset.points.nonEmpty, "Cannot train on empty dataset")
      buildTree(dataset, 0)
    }
    
    private def buildTree(dataset: Dataset, currentDepth: Int): TreeNode = {
      val labels = dataset.points.map(_.label)
      val majorityLabel = labels.groupBy(identity).maxBy(_._2.size)._1
      
      // Stopping criteria
      if (currentDepth >= maxDepth || 
          dataset.size < minSamplesSplit || 
          dataset.size <= minSamplesLeaf ||
          labels.distinct.size == 1) {
        return LeafNode(majorityLabel)
      }
      
      // Find best split
      findBestSplit(dataset) match {
        case Some((featureIndex, threshold, leftDataset, rightDataset)) =>
          val leftChild = buildTree(leftDataset, currentDepth + 1)
          val rightChild = buildTree(rightDataset, currentDepth + 1)
          InternalNode(featureIndex, threshold, leftChild, rightChild)
        case None =>
          LeafNode(majorityLabel)
      }
    }
    
    private def findBestSplit(dataset: Dataset): Option[(Int, Double, Dataset, Dataset)] = {
      val featureIndices = (0 until dataset.featureCount).toVector
      
      val splits = for {
        featureIndex <- featureIndices
        threshold <- generateThresholds(dataset, featureIndex)
        (left, right) = dataset.splitByFeature(featureIndex, threshold)
        if left.size >= minSamplesLeaf && right.size >= minSamplesLeaf
      } yield {
        val gain = informationGain(dataset, left, right)
        (gain, featureIndex, threshold, left, right)
      }
      
      splits.maxByOption(_._1).map {
        case (_, featureIndex, threshold, left, right) => 
          (featureIndex, threshold, left, right)
      }
    }
    
    private def generateThresholds(dataset: Dataset, featureIndex: Int): Vector[Double] = {
      val values = dataset.points.map(_.features(featureIndex)).distinct.sorted
      values.zip(values.tail).map { case (a, b) => (a + b) / 2 }
    }
    
    private def informationGain(parent: Dataset, left: Dataset, right: Dataset): Double = {
      val totalSize = parent.size.toDouble
      val leftWeight = left.size / totalSize
      val rightWeight = right.size / totalSize
      
      entropy(parent) - (leftWeight * entropy(left) + rightWeight * entropy(right))
    }
    
    private def entropy(dataset: Dataset): Double = {
      val labelCounts = dataset.points.groupBy(_.label).mapValues(_.size.toDouble)
      val total = dataset.size.toDouble
      
      -labelCounts.values.map { count =>
        val p = count / total
        if (p > 0) p * log(p) / log(2) else 0.0
      }.sum
    }
  }
  
  // Random Forest configuration
  case class RandomForestConfig(
    nTrees: Int = 100,
    maxDepth: Int = 10,
    minSamplesLeaf: Int = 1,
    minSamplesSplit: Int = 2,
    maxFeatures: Option[Int] = None,
    bootstrap: Boolean = true,
    oobScore: Boolean = false,
    randomSeed: Long = 42L
  ) {
    require(nTrees > 0, "Number of trees must be positive")
    require(maxDepth > 0, "Max depth must be positive")
    require(minSamplesLeaf > 0, "Min samples leaf must be positive")
    require(minSamplesSplit >= 2, "Min samples split must be at least 2")
  }
  
  // Actor for parallel tree training
  class TreeTrainerActor extends Actor {
    implicit val random: Random = new Random()
    
    def receive = {
      case TrainTreeMessage(dataset, config, treeId) =>
        val result = Try {
          val tree = new DecisionTree(config.maxDepth, config.minSamplesLeaf, config.minSamplesSplit)
          
          // Prepare training data (with bootstrap sampling if enabled)
          val trainingDataset = if (config.bootstrap) {
            dataset.bootstrap()
          } else {
            dataset
          }
          
          // Feature subset selection for bagging
          val featureSubset = config.maxFeatures.map { count =>
            dataset.randomFeatureSubset(math.min(count, dataset.featureCount))
          }.getOrElse((0 until dataset.featureCount).toVector)
          
          // Train tree
          val trainedTree = tree.train(trainingDataset).get
          
          // Calculate metrics
          val trainingAccuracy = calculateAccuracy(trainedTree, trainingDataset)
          val oobAccuracy = if (config.bootstrap && config.oobScore) {
            val oobSamples = dataset.points.diff(trainingDataset.points)
            if (oobSamples.nonEmpty) {
              Some(calculateAccuracy(trainedTree, Dataset(oobSamples)))
            } else None
          } else None
          
          TrainedTreeResult(treeId, trainedTree, TreeMetrics(
            trainedTree.depth,
            trainedTree.leafCount,
            trainingAccuracy,
            oobAccuracy
          ))
        }
        
        sender() ! result
    }
    
    private def calculateAccuracy(tree: TreeNode, dataset: Dataset): Double = {
      val correct = dataset.points.count { point =>
        val prediction = tree.predict(point.features)
        math.abs(prediction - point.label) < 0.5 // For classification
      }
      correct.toDouble / dataset.size
    }
  }
  
  // Messages for actor communication
  case class TrainTreeMessage(dataset: Dataset, config: RandomForestConfig, treeId: Int)
  case class TrainedTreeResult(treeId: Int, tree: TreeNode, metrics: TreeMetrics)
  
  // Main Random Forest class
  class RandomForest(config: RandomForestConfig = RandomForestConfig()) {
    private implicit val system: ActorSystem = ActorSystem("RandomForestSystem")
    private implicit val ec: ExecutionContext = system.dispatcher
    private implicit val timeout: Timeout = Timeout(30.seconds)
    
    private var trainedTrees: Vector[TreeNode] = Vector.empty
    private var forestMetrics: Option[ForestMetrics] = None
    
    def train(dataset: Dataset): Try[Unit] = Try {
      require(dataset.points.nonEmpty, "Cannot train on empty dataset")
      require(dataset.points.forall(_.features.length == dataset.featureCount), 
              "All data points must have the same number of features")
      
      println(s"🌲 Training Random Forest with ${config.nTrees} trees...")
      println(s"📊 Dataset: ${dataset.size} samples, ${dataset.featureCount} features")
      
      // Create tree trainer actors
      val trainers = (1 to config.nTrees).map { _ =>
        system.actorOf(Props[TreeTrainerActor])
      }
      
      // Send training messages
      val trainingFutures = trainers.zipWithIndex.map { case (trainer, treeId) =>
        (trainer ? TrainTreeMessage(dataset, config, treeId)).mapTo[Try[TrainedTreeResult]]
      }
      
      // Wait for all trees to be trained
      val trainingResults = Await.result(Future.sequence(trainingFutures), 60.seconds)
      
      // Process results
      val successfulTrees = trainingResults.collect {
        case Success(result) => result
      }.sortBy(_.treeId)
      
      val failedCount = trainingResults.count(_.isFailure)
      if (failedCount > 0) {
        println(s"⚠️ Warning: ${failedCount} trees failed to train")
      }
      
      trainedTrees = successfulTrees.map(_.tree).toVector
      
      // Calculate forest metrics
      val treeMetrics = successfulTrees.map(_.metrics)
      val avgDepth = treeMetrics.map(_.depth.toDouble).sum / treeMetrics.size
      val avgLeaves = treeMetrics.map(_.leafCount.toDouble).sum / treeMetrics.size
      val avgAccuracy = treeMetrics.map(_.trainingAccuracy).sum / treeMetrics.size
      val oobErrors = treeMetrics.flatMap(_.oobError)
      val avgOobError = if (oobErrors.nonEmpty) Some(oobErrors.sum / oobErrors.size) else None
      
      forestMetrics = Some(ForestMetrics(
        nTrees = trainedTrees.size,
        avgDepth = avgDepth,
        avgLeafCount = avgLeaves,
        avgTrainingAccuracy = avgAccuracy,
        oobError = avgOobError
      ))
      
      println(s"✅ Successfully trained ${trainedTrees.size} trees")
      println(s"📈 Average depth: ${avgDepth.formatted("%.2f")}")
      println(s"📈 Average leaves: ${avgLeaves.formatted("%.2f")}")
      println(s"📈 Average training accuracy: ${(avgAccuracy * 100).formatted("%.2f")}%")
      avgOobError.foreach { oob =>
        println(s"📈 Out-of-bag error: ${(oob * 100).formatted("%.2f")}%")
      }
      
      // Shutdown actors
      trainers.foreach(system.stop)
      
    }.recover {
      case e: Exception => 
        system.terminate()
        throw TrainingError(s"Random Forest training failed: ${e.getMessage}")
    }
    
    def predict(features: Vector[Double]): Try[Double] = Try {
      require(trainedTrees.nonEmpty, "Model must be trained before prediction")
      require(features.length == trainedTrees.head.predict(features); features.length, 
              "Feature count mismatch")
      
      // Get predictions from all trees
      val predictions = trainedTrees.map(_.predict(features))
      
      // For regression: average predictions
      // For classification: majority vote
      if (isClassification) {
        // Majority vote
        predictions.groupBy(identity).maxBy(_._2.size)._1
      } else {
        // Average for regression
        predictions.sum / predictions.size
      }
    }.recover {
      case e: Exception => 
        throw PredictionError(s"Prediction failed: ${e.getMessage}")
    }
    
    def predictBatch(featuresVector: Vector[Vector[Double]]): Try[Vector[Double]] = Try {
      featuresVector.map(features => predict(features).get)
    }
    
    def evaluate(testDataset: Dataset): Try[EvaluationMetrics] = Try {
      require(trainedTrees.nonEmpty, "Model must be trained before evaluation")
      
      val predictions = testDataset.points.map { point =>
        predict(point.features).get
      }
      
      val trueLabels = testDataset.points.map(_.label)
      
      // Calculate metrics
      val accuracy = calculateAccuracy(trueLabels, predictions)
      val mse = calculateMSE(trueLabels, predictions)
      val mae = calculateMAE(trueLabels, predictions)
      
      EvaluationMetrics(accuracy, mse, mae, testDataset.size)
      
    }.recover {
      case e: Exception =>
        throw ValidationError(s"Evaluation failed: ${e.getMessage}")
    }
    
    def featureImportance(): Option[Vector[Double]] = {
      if (trainedTrees.isEmpty) return None
      
      val featureCount = trainedTrees.head match {
        case InternalNode(_, _, _, _) => getFeatureCount(trainedTrees.head)
        case _ => 0
      }
      
      if (featureCount == 0) return None
      
      val importances = Array.fill(featureCount)(0.0)
      
      trainedTrees.foreach { tree =>
        calculateFeatureImportance(tree, importances)
      }
      
      // Normalize
      val total = importances.sum
      if (total > 0) {
        Some(importances.map(_ / total).toVector)
      } else {
        Some(Vector.fill(featureCount)(1.0 / featureCount))
      }
    }
    
    def getMetrics: Option[ForestMetrics] = forestMetrics
    
    def shutdown(): Unit = {
      system.terminate()
    }
    
    // Helper methods
    private def isClassification: Boolean = {
      // Simple heuristic: if all predictions are integers, assume classification
      trainedTrees.nonEmpty && {
        val samplePredictions = trainedTrees.take(5).map(_.predict(Vector.fill(1)(0.0)))
        samplePredictions.forall(p => p == p.round.toDouble)
      }
    }
    
    private def calculateAccuracy(trueLabels: Vector[Double], predictions: Vector[Double]): Double = {
      val correct = trueLabels.zip(predictions).count { case (true_val, pred) =>
        math.abs(true_val - pred) < 0.5
      }
      correct.toDouble / trueLabels.size
    }
    
    private def calculateMSE(trueLabels: Vector[Double], predictions: Vector[Double]): Double = {
      val squaredErrors = trueLabels.zip(predictions).map { case (true_val, pred) =>
        pow(true_val - pred, 2)
      }
      squaredErrors.sum / squaredErrors.size
    }
    
    private def calculateMAE(trueLabels: Vector[Double], predictions: Vector[Double]): Double = {
      val absoluteErrors = trueLabels.zip(predictions).map { case (true_val, pred) =>
        math.abs(true_val - pred)
      }
      absoluteErrors.sum / absoluteErrors.size
    }
    
    private def getFeatureCount(tree: TreeNode): Int = tree match {
      case InternalNode(featureIndex, _, left, right) =>
        math.max(featureIndex + 1, math.max(getFeatureCount(left), getFeatureCount(right)))
      case LeafNode(_) => 0
    }
    
    private def calculateFeatureImportance(tree: TreeNode, importances: Array[Double]): Unit = tree match {
      case InternalNode(featureIndex, _, left, right) =>
        importances(featureIndex) += 1.0
        calculateFeatureImportance(left, importances)
        calculateFeatureImportance(right, importances)
      case LeafNode(_) => // No importance for leaves
    }
  }
  
  // Metrics case classes
  case class ForestMetrics(
    nTrees: Int,
    avgDepth: Double,
    avgLeafCount: Double,
    avgTrainingAccuracy: Double,
    oobError: Option[Double]
  )
  
  case class EvaluationMetrics(
    accuracy: Double,
    mse: Double,
    mae: Double,
    sampleCount: Int
  )
  
  // Data generation utilities
  object DataUtils {
    
    def generateClassificationDataset(
      samples: Int = 1000,
      features: Int = 4,
      classes: Int = 2,
      noise: Double = 0.1,
      randomSeed: Long = 42L
    ): Dataset = {
      implicit val random = new Random(randomSeed)
      
      val points = (1 to samples).map { _ =>
        val featureVector = Vector.fill(features)(random.nextGaussian())
        
        // Simple linear combination for class determination
        val linearCombination = featureVector.sum + random.nextGaussian() * noise
        val classLabel = if (classes == 2) {
          if (linearCombination > 0) 1.0 else 0.0
        } else {
          (linearCombination % classes).abs.toDouble
        }
        
        DataPoint(featureVector, classLabel)
      }.toVector
      
      Dataset(points)
    }
    
    def generateRegressionDataset(
      samples: Int = 1000,
      features: Int = 4,
      noise: Double = 0.1,
      randomSeed: Long = 42L
    ): Dataset = {
      implicit val random = new Random(randomSeed)
      
      // Generate random coefficients
      val coefficients = Vector.fill(features)(random.nextGaussian())
      
      val points = (1 to samples).map { _ =>
        val featureVector = Vector.fill(features)(random.nextGaussian() * 2)
        
        // Linear combination with noise
        val target = featureVector.zip(coefficients).map { case (f, c) => f * c }.sum + 
                    random.nextGaussian() * noise
        
        DataPoint(featureVector, target)
      }.toVector
      
      Dataset(points)
    }
    
    def trainTestSplit(dataset: Dataset, testRatio: Double = 0.2, randomSeed: Long = 42L): (Dataset, Dataset) = {
      implicit val random = new Random(randomSeed)
      
      val shuffled = random.shuffle(dataset.points)
      val splitPoint = (dataset.size * (1 - testRatio)).toInt
      
      val trainPoints = shuffled.take(splitPoint)
      val testPoints = shuffled.drop(splitPoint)
      
      (Dataset(trainPoints), Dataset(testPoints))
    }
    
    def generateIrisLikeDataset(samples: Int = 150, randomSeed: Long = 42L): Dataset = {
      implicit val random = new Random(randomSeed)
      
      val points = (1 to samples).map { i =>
        val classId = i % 3
        val baseFeatures = classId match {
          case 0 => Vector(5.0, 3.5, 1.5, 0.2)  // Setosa-like
          case 1 => Vector(6.0, 3.0, 4.0, 1.3)  // Versicolor-like  
          case _ => Vector(7.0, 3.2, 5.5, 2.0)  // Virginica-like
        }
        
        val noisyFeatures = baseFeatures.map(_ + random.nextGaussian() * 0.3)
        DataPoint(noisyFeatures, classId.toDouble)
      }.toVector
      
      Dataset(random.shuffle(points))
    }
  }
  
  // Demo application
  object Demo {
    def main(args: Array[String]): Unit = {
      println("🌲 Scala Random Forest Implementation Demo")
      println("=" * 50)
      
      try {
        // Generate synthetic dataset
        println("📊 Generating synthetic classification dataset...")
        val fullDataset = DataUtils.generateIrisLikeDataset(samples = 300)
        val (trainDataset, testDataset) = DataUtils.trainTestSplit(fullDataset, testRatio = 0.3)
        
        println(s"📈 Train samples: ${trainDataset.size}, Test samples: ${testDataset.size}")
        println(s"📈 Features: ${fullDataset.featureCount}")
        
        // Configure Random Forest
        val config = RandomForestConfig(
          nTrees = 50,
          maxDepth = 10,
          minSamplesLeaf = 2,
          minSamplesSplit = 5,
          maxFeatures = Some(2),
          bootstrap = true,
          oobScore = true
        )
        
        println(s"\n🏗️ Building Random Forest with configuration:")
        println(s"   Trees: ${config.nTrees}")
        println(s"   Max Depth: ${config.maxDepth}")
        println(s"   Max Features: ${config.maxFeatures.getOrElse("all")}")
        println(s"   Bootstrap: ${config.bootstrap}")
        println(s"   OOB Score: ${config.oobScore}")
        
        // Train Random Forest
        val forest = new RandomForest(config)
        val startTime = System.currentTimeMillis()
        
        forest.train(trainDataset) match {
          case Success(_) =>
            val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0
            println(s"\n✅ Training completed in ${trainingTime.formatted("%.2f")} seconds")
            
            // Display forest metrics
            forest.getMetrics.foreach { metrics =>
              println(s"\n📊 Random Forest Metrics:")
              println(s"   Trees trained: ${metrics.nTrees}")
              println(s"   Average tree depth: ${metrics.avgDepth.formatted("%.2f")}")
              println(s"   Average leaf count: ${metrics.avgLeafCount.formatted("%.2f")}")
              println(s"   Training accuracy: ${(metrics.avgTrainingAccuracy * 100).formatted("%.2f")}%")
              metrics.oobError.foreach { oob =>
                println(s"   Out-of-bag error: ${(oob * 100).formatted("%.2f")}%")
              }
            }
            
            // Feature importance
            forest.featureImportance().foreach { importances =>
              println(s"\n🎯 Feature Importance:")
              importances.zipWithIndex.foreach { case (importance, idx) =>
                println(s"   Feature ${idx}: ${(importance * 100).formatted("%.2f")}%")
              }
            }
            
            // Evaluate on test set
            println(s"\n🔬 Evaluating on test set...")
            forest.evaluate(testDataset) match {
              case Success(metrics) =>
                println(s"📈 Test Accuracy: ${(metrics.accuracy * 100).formatted("%.2f")}%")
                println(s"📈 Test MSE: ${metrics.mse.formatted("%.6f")}")
                println(s"📈 Test MAE: ${metrics.mae.formatted("%.6f")}")
                
                // Test individual predictions
                println(s"\n🔮 Sample Predictions:")
                testDataset.points.take(5).foreach { point =>
                  forest.predict(point.features) match {
                    case Success(prediction) =>
                      println(s"   Features: ${point.features.map(_.formatted("%.2f")).mkString("[", ", ", "]")} → " +
                              s"Predicted: ${prediction.formatted("%.0f")}, Actual: ${point.label.formatted("%.0f")}")
                    case Failure(e) =>
                      println(s"   Prediction failed: ${e.getMessage}")
                  }
                }
                
                println(s"\n✅ Random Forest demonstration completed successfully!")
                
              case Failure(e) =>
                println(s"❌ Evaluation failed: ${e.getMessage}")
            }
            
            // Clean shutdown
            forest.shutdown()
            
          case Failure(e) =>
            println(s"❌ Training failed: ${e.getMessage}")
            forest.shutdown()
        }
        
      } catch {
        case e: Exception =>
          println(s"❌ Demo failed: ${e.getMessage}")
          e.printStackTrace()
      }
    }
  }
}

// ScalaTest test suite example
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll

class RandomForestSpec extends AnyFlatSpec with Matchers with BeforeAndAfterAll {
  import RandomForestML._
  
  "A Dataset" should "validate input correctly" in {
    val points = Vector(
      DataPoint(Vector(1.0, 2.0), 0.0),
      DataPoint(Vector(2.0, 3.0), 1.0)
    )
    val dataset = Dataset(points)
    
    dataset.size should be(2)
    dataset.featureCount should be(2)
  }
  
  it should "split by feature correctly" in {
    val dataset = DataUtils.generateClassificationDataset(samples = 100)
    val (left, right) = dataset.splitByFeature(0, 0.0)
    
    (left.size + right.size) should be(dataset.size)
    left.points.foreach(_.features(0) should be <= 0.0)
    right.points.foreach(_.features(0) should be > 0.0)
  }
  
  "A DecisionTree" should "train successfully on valid data" in {
    implicit val random = new Random(42)
    val tree = new DecisionTree()
    val dataset = DataUtils.generateClassificationDataset(samples = 100)
    
    tree.train(dataset) should be a 'success
  }
  
  "A RandomForest" should "train and predict correctly" in {
    val config = RandomForestConfig(nTrees = 10, maxDepth = 5)
    val forest = new RandomForest(config)
    val dataset = DataUtils.generateClassificationDataset(samples = 200)
    
    forest.train(dataset) should be a 'success
    
    val testPoint = dataset.points.head
    forest.predict(testPoint.features) should be a 'success
    
    forest.shutdown()
  }
  
  it should "calculate feature importance correctly" in {
    val config = RandomForestConfig(nTrees = 5, maxDepth = 3)
    val forest = new RandomForest(config)
    val dataset = DataUtils.generateClassificationDataset(samples = 100, features = 3)
    
    forest.train(dataset)
    val importance = forest.featureImportance()
    
    importance should not be None
    importance.get.size should be(3)
    importance.get.sum should be(1.0 +- 0.01)
    
    forest.shutdown()
  }
  
  "DataUtils" should "generate valid classification data" in {
    val dataset = DataUtils.generateClassificationDataset(samples = 50, features = 3, classes = 2)
    
    dataset.size should be(50)
    dataset.featureCount should be(3)
    dataset.points.foreach { point =>
      point.label should (be(0.0) or be(1.0))
      point.features.size should be(3)
    }
  }
  
  it should "split data correctly" in {
    val dataset = DataUtils.generateClassificationDataset(samples = 100)
    val (train, test) = DataUtils.trainTestSplit(dataset, testRatio = 0.3)
    
    train.size should be(70)
    test.size should be(30)
    (train.points ++ test.points).size should be(dataset.size)
  }
}

// SBT build configuration comment
/*
// build.sbt
name := "RandomForestML"
version := "1.0"
scalaVersion := "2.13.8"

libraryDependencies ++= Seq(
  "com.typesafe.akka" %% "akka-actor-typed" % "2.6.19",
  "org.scalatest" %% "scalatest" % "3.2.12" % Test
)
*/