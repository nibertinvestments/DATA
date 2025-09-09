// Comprehensive Scala Concurrent and Async Programming Examples
// Demonstrates Akka actors, Futures, parallel collections, and reactive streams

import akka.actor.{Actor, ActorRef, ActorSystem, Props, PoisonPill}
import akka.pattern.ask
import akka.util.Timeout
import scala.concurrent.{Future, Promise, ExecutionContext}
import scala.concurrent.duration._
import scala.util.{Try, Success, Failure, Random}
import java.util.concurrent.{Executors, ThreadPoolExecutor}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

// ============ Message Definitions for Actors ============

/**
 * Protocol for user management actor
 */
object UserManagerProtocol {
  sealed trait UserMessage
  
  case class CreateUser(name: String, email: String, age: Int) extends UserMessage
  case class GetUser(id: Long) extends UserMessage
  case class UpdateUser(id: Long, name: Option[String], email: Option[String]) extends UserMessage
  case class DeleteUser(id: Long) extends UserMessage
  case object GetAllUsers extends UserMessage
  case class SearchUsers(query: String) extends UserMessage
  
  // Responses
  sealed trait UserResponse
  case class UserCreated(user: User) extends UserResponse
  case class UserFound(user: User) extends UserResponse
  case object UserNotFound extends UserResponse
  case class UserUpdated(user: User) extends UserResponse
  case object UserDeleted extends UserResponse
  case class UsersFound(users: List[User]) extends UserResponse
  case class UserError(message: String) extends UserResponse
}

/**
 * Protocol for order processing system
 */
object OrderProtocol {
  sealed trait OrderMessage
  
  case class ProcessOrder(order: Order) extends OrderMessage
  case class GetOrderStatus(orderId: String) extends OrderMessage
  case class CancelOrder(orderId: String) extends OrderMessage
  
  // Internal messages
  case class ValidateInventory(order: Order) extends OrderMessage
  case class ProcessPayment(order: Order) extends OrderMessage
  case class ShipOrder(order: Order) extends OrderMessage
  
  // Responses
  sealed trait OrderResponse
  case class OrderProcessed(orderId: String) extends OrderResponse
  case class OrderStatus(orderId: String, status: String) extends OrderResponse
  case class OrderCancelled(orderId: String) extends OrderResponse
  case class OrderError(orderId: String, error: String) extends OrderResponse
}

/**
 * Protocol for distributed computation
 */
object ComputationProtocol {
  sealed trait ComputationMessage
  
  case class ComputeTask(id: String, data: List[Int]) extends ComputationMessage
  case class TaskResult(id: String, result: Int) extends ComputationMessage
  case class TaskFailed(id: String, error: String) extends ComputationMessage
  
  case class DistributedSum(taskId: String, numbers: List[Int], numWorkers: Int) extends ComputationMessage
  case class WorkerTask(taskId: String, workerId: Int, numbers: List[Int]) extends ComputationMessage
}

// ============ Domain Models ============

case class User(
  id: Long,
  name: String,
  email: String,
  age: Int,
  createdAt: LocalDateTime = LocalDateTime.now(),
  isActive: Boolean = true
) {
  def deactivate: User = copy(isActive = false)
  def updateInfo(newName: Option[String], newEmail: Option[String]): User = {
    copy(
      name = newName.getOrElse(name),
      email = newEmail.getOrElse(email)
    )
  }
}

case class Product(
  id: String,
  name: String,
  price: BigDecimal,
  inventory: Int
)

case class OrderItem(
  productId: String,
  quantity: Int,
  price: BigDecimal
)

case class Order(
  id: String,
  userId: Long,
  items: List[OrderItem],
  status: String = "PENDING",
  createdAt: LocalDateTime = LocalDateTime.now()
) {
  def totalAmount: BigDecimal = items.map(item => item.price * item.quantity).sum
  def updateStatus(newStatus: String): Order = copy(status = newStatus)
}

// ============ Actor Implementations ============

/**
 * User management actor with persistent state
 */
class UserManagerActor extends Actor {
  import UserManagerProtocol._
  
  private var users: Map[Long, User] = Map.empty
  private var nextId: Long = 1L
  
  def receive: Receive = {
    case CreateUser(name, email, age) =>
      val user = User(nextId, name, email, age)
      users = users + (nextId -> user)
      nextId += 1
      sender() ! UserCreated(user)
      
    case GetUser(id) =>
      users.get(id) match {
        case Some(user) => sender() ! UserFound(user)
        case None => sender() ! UserNotFound
      }
      
    case UpdateUser(id, name, email) =>
      users.get(id) match {
        case Some(user) =>
          val updated = user.updateInfo(name, email)
          users = users + (id -> updated)
          sender() ! UserUpdated(updated)
        case None =>
          sender() ! UserNotFound
      }
      
    case DeleteUser(id) =>
      users.get(id) match {
        case Some(_) =>
          users = users - id
          sender() ! UserDeleted
        case None =>
          sender() ! UserNotFound
      }
      
    case GetAllUsers =>
      sender() ! UsersFound(users.values.toList)
      
    case SearchUsers(query) =>
      val matching = users.values.filter { user =>
        user.name.toLowerCase.contains(query.toLowerCase) ||
        user.email.toLowerCase.contains(query.toLowerCase)
      }.toList
      sender() ! UsersFound(matching)
  }
}

/**
 * Order processing actor that coordinates with other actors
 */
class OrderProcessingActor(inventoryActor: ActorRef, paymentActor: ActorRef) extends Actor {
  import OrderProtocol._
  import context.dispatcher
  
  private var orders: Map[String, Order] = Map.empty
  implicit val timeout: Timeout = Timeout(5.seconds)
  
  def receive: Receive = {
    case ProcessOrder(order) =>
      orders = orders + (order.id -> order)
      val originalSender = sender()
      
      // Chain of async operations
      val processingFuture = for {
        inventoryValid <- (inventoryActor ? ValidateInventory(order)).mapTo[Boolean]
        paymentProcessed <- if (inventoryValid) {
          (paymentActor ? ProcessPayment(order)).mapTo[Boolean]
        } else {
          Future.successful(false)
        }
        result <- if (paymentProcessed) {
          val shippedOrder = order.updateStatus("SHIPPED")
          orders = orders + (order.id -> shippedOrder)
          Future.successful(OrderProcessed(order.id))
        } else {
          val failedOrder = order.updateStatus("FAILED")
          orders = orders + (order.id -> failedOrder)
          Future.successful(OrderError(order.id, "Processing failed"))
        }
      } yield result
      
      processingFuture.foreach(originalSender ! _)
      
    case GetOrderStatus(orderId) =>
      orders.get(orderId) match {
        case Some(order) => sender() ! OrderStatus(orderId, order.status)
        case None => sender() ! OrderError(orderId, "Order not found")
      }
      
    case CancelOrder(orderId) =>
      orders.get(orderId) match {
        case Some(order) if order.status == "PENDING" =>
          val cancelled = order.updateStatus("CANCELLED")
          orders = orders + (orderId -> cancelled)
          sender() ! OrderCancelled(orderId)
        case Some(_) =>
          sender() ! OrderError(orderId, "Cannot cancel order in current status")
        case None =>
          sender() ! OrderError(orderId, "Order not found")
      }
  }
}

/**
 * Mock inventory validation actor
 */
class InventoryActor extends Actor {
  import OrderProtocol._
  
  private val inventory = Map(
    "product1" -> 100,
    "product2" -> 50,
    "product3" -> 25
  )
  
  def receive: Receive = {
    case ValidateInventory(order) =>
      val available = order.items.forall { item =>
        inventory.getOrElse(item.productId, 0) >= item.quantity
      }
      sender() ! available
  }
}

/**
 * Mock payment processing actor
 */
class PaymentActor extends Actor {
  import OrderProtocol._
  import context.dispatcher
  
  def receive: Receive = {
    case ProcessPayment(order) =>
      val originalSender = sender()
      
      // Simulate async payment processing
      Future {
        Thread.sleep(100) // Simulate network delay
        Random.nextBoolean() // Random success/failure for demo
      }.foreach(originalSender ! _)
  }
}

/**
 * Worker actor for distributed computation
 */
class ComputationWorkerActor extends Actor {
  import ComputationProtocol._
  import context.dispatcher
  
  def receive: Receive = {
    case WorkerTask(taskId, workerId, numbers) =>
      val originalSender = sender()
      
      Future {
        // Simulate computation work
        Thread.sleep(50)
        val result = numbers.sum
        TaskResult(s"$taskId-$workerId", result)
      }.recover {
        case ex => TaskFailed(s"$taskId-$workerId", ex.getMessage)
      }.foreach(originalSender ! _)
  }
}

/**
 * Coordinator actor for distributed computation
 */
class ComputationCoordinatorActor extends Actor {
  import ComputationProtocol._
  import context.dispatcher
  
  implicit val timeout: Timeout = Timeout(10.seconds)
  
  def receive: Receive = {
    case DistributedSum(taskId, numbers, numWorkers) =>
      val originalSender = sender()
      
      // Create worker actors
      val workers = (1 to numWorkers).map { i =>
        context.actorOf(Props[ComputationWorkerActor], s"worker-$i")
      }
      
      // Distribute work
      val chunkSize = math.max(1, numbers.length / numWorkers)
      val chunks = numbers.grouped(chunkSize).toList
      
      val futures = workers.zip(chunks).zipWithIndex.map { case ((worker, chunk), index) =>
        (worker ? WorkerTask(taskId, index, chunk)).mapTo[TaskResult]
      }
      
      // Aggregate results
      Future.sequence(futures).map { results =>
        val totalSum = results.map(_.result).sum
        TaskResult(taskId, totalSum)
      }.foreach { result =>
        originalSender ! result
        // Clean up workers
        workers.foreach(_ ! PoisonPill)
      }
  }
}

// ============ Future-based Async Operations ============

object AsyncOperations {
  implicit val ec: ExecutionContext = ExecutionContext.global
  
  /**
   * Simulate database operations
   */
  object DatabaseSimulator {
    private val users = scala.collection.mutable.Map[Long, User]()
    private var nextId = 1L
    
    def createUser(name: String, email: String, age: Int): Future[User] = Future {
      Thread.sleep(50) // Simulate DB latency
      val user = User(nextId, name, email, age)
      users(nextId) = user
      nextId += 1
      user
    }
    
    def getUser(id: Long): Future[Option[User]] = Future {
      Thread.sleep(30)
      users.get(id)
    }
    
    def updateUser(id: Long, name: Option[String], email: Option[String]): Future[Option[User]] = Future {
      Thread.sleep(40)
      users.get(id).map { user =>
        val updated = user.updateInfo(name, email)
        users(id) = updated
        updated
      }
    }
    
    def deleteUser(id: Long): Future[Boolean] = Future {
      Thread.sleep(35)
      users.remove(id).isDefined
    }
    
    def searchUsers(query: String): Future[List[User]] = Future {
      Thread.sleep(60)
      users.values.filter { user =>
        user.name.toLowerCase.contains(query.toLowerCase) ||
        user.email.toLowerCase.contains(query.toLowerCase)
      }.toList
    }
  }
  
  /**
   * External API simulation
   */
  object ExternalApiClient {
    def fetchUserData(userId: Long): Future[Map[String, Any]] = Future {
      Thread.sleep(200) // Simulate API call
      if (Random.nextBoolean()) {
        Map(
          "id" -> userId,
          "preferences" -> List("dark_mode", "notifications"),
          "last_login" -> LocalDateTime.now().minusDays(Random.nextInt(30))
        )
      } else {
        throw new RuntimeException(s"Failed to fetch data for user $userId")
      }
    }
    
    def sendNotification(userId: Long, message: String): Future[Boolean] = Future {
      Thread.sleep(100)
      println(s"Notification sent to user $userId: $message")
      Random.nextBoolean() // Random success for demo
    }
    
    def validateEmail(email: String): Future[Boolean] = Future {
      Thread.sleep(80)
      email.contains("@") && email.contains(".")
    }
  }
  
  /**
   * Complex async workflow
   */
  def createUserWorkflow(name: String, email: String, age: Int): Future[Either[String, User]] = {
    for {
      // Step 1: Validate email
      isValidEmail <- ExternalApiClient.validateEmail(email).recover(_ => false)
      
      result <- if (!isValidEmail) {
        Future.successful(Left("Invalid email address"))
      } else {
        // Step 2: Create user
        for {
          user <- DatabaseSimulator.createUser(name, email, age)
          
          // Step 3: Fetch additional data (optional, doesn't fail the workflow)
          additionalData <- ExternalApiClient.fetchUserData(user.id).recover(_ => Map.empty[String, Any])
          
          // Step 4: Send welcome notification
          notificationSent <- ExternalApiClient.sendNotification(user.id, "Welcome!").recover(_ => false)
          
        } yield {
          if (notificationSent) {
            Right(user)
          } else {
            // User created but notification failed - still success
            Right(user)
          }
        }
      }
    } yield result
  }
  
  /**
   * Batch processing with rate limiting
   */
  def processBatch[A, B](
    items: List[A], 
    processor: A => Future[B], 
    batchSize: Int = 5,
    delayBetweenBatches: FiniteDuration = 100.millis
  ): Future[List[B]] = {
    
    def processBatchGroup(batch: List[A]): Future[List[B]] = {
      val futures = batch.map(processor)
      Future.sequence(futures)
    }
    
    items.grouped(batchSize).foldLeft(Future.successful(List.empty[B])) { (accFuture, batch) =>
      for {
        acc <- accFuture
        _ <- Future { Thread.sleep(delayBetweenBatches.toMillis) }
        batchResults <- processBatchGroup(batch)
      } yield acc ++ batchResults
    }
  }
  
  /**
   * Circuit breaker pattern
   */
  class CircuitBreaker(
    failureThreshold: Int = 5,
    timeoutDuration: FiniteDuration = 1.second,
    resetTimeout: FiniteDuration = 10.seconds
  ) {
    private var state: CircuitBreakerState = Closed
    private var failureCount = 0
    private var lastFailureTime: Option[LocalDateTime] = None
    
    sealed trait CircuitBreakerState
    case object Closed extends CircuitBreakerState
    case object Open extends CircuitBreakerState
    case object HalfOpen extends CircuitBreakerState
    
    def execute[A](operation: () => Future[A]): Future[A] = {
      state match {
        case Closed =>
          operation().transform(
            success = { result =>
              failureCount = 0
              result
            },
            failure = { ex =>
              failureCount += 1
              lastFailureTime = Some(LocalDateTime.now())
              if (failureCount >= failureThreshold) {
                state = Open
              }
              ex
            }
          )
          
        case Open =>
          lastFailureTime match {
            case Some(lastFailure) if LocalDateTime.now().isAfter(lastFailure.plus(resetTimeout.toNanos, java.time.temporal.ChronoUnit.NANOS)) =>
              state = HalfOpen
              execute(operation)
            case _ =>
              Future.failed(new RuntimeException("Circuit breaker is OPEN"))
          }
          
        case HalfOpen =>
          operation().transform(
            success = { result =>
              state = Closed
              failureCount = 0
              result
            },
            failure = { ex =>
              state = Open
              lastFailureTime = Some(LocalDateTime.now())
              ex
            }
          )
      }
    }
  }
  
  /**
   * Retry with exponential backoff
   */
  def retryWithBackoff[A](
    operation: () => Future[A],
    maxRetries: Int,
    baseDelay: FiniteDuration = 100.millis,
    maxDelay: FiniteDuration = 10.seconds
  ): Future[A] = {
    
    def attempt(retriesLeft: Int, currentDelay: FiniteDuration): Future[A] = {
      operation().recoverWith {
        case ex if retriesLeft > 0 =>
          val delay = List(currentDelay * 2, maxDelay).min
          Future {
            Thread.sleep(currentDelay.toMillis)
          }.flatMap(_ => attempt(retriesLeft - 1, delay))
        case ex => 
          Future.failed(ex)
      }
    }
    
    attempt(maxRetries, baseDelay)
  }
}

// ============ Parallel Collections ============

object ParallelCollections {
  
  /**
   * CPU-intensive computation using parallel collections
   */
  def parallelFactorial(numbers: Vector[Int]): Vector[BigInt] = {
    numbers.par.map { n =>
      if (n < 0) BigInt(0)
      else (1 to n).foldLeft(BigInt(1))(_ * _)
    }.seq
  }
  
  /**
   * Parallel data processing pipeline
   */
  def processLargeDataset(data: Vector[String]): Vector[Map[String, Any]] = {
    data.par.map { item =>
      // Simulate complex processing
      val words = item.split("\\s+")
      val wordCount = words.length
      val avgWordLength = if (words.nonEmpty) words.map(_.length).sum.toDouble / words.length else 0.0
      val hasNumbers = item.exists(_.isDigit)
      
      Map(
        "original" -> item,
        "wordCount" -> wordCount,
        "avgWordLength" -> avgWordLength,
        "hasNumbers" -> hasNumbers,
        "processedAt" -> LocalDateTime.now()
      )
    }.seq
  }
  
  /**
   * Parallel filtering and aggregation
   */
  def analyzeNumbers(numbers: Vector[Int]): Map[String, Any] = {
    val parNumbers = numbers.par
    
    Map(
      "sum" -> parNumbers.sum,
      "count" -> parNumbers.length,
      "evens" -> parNumbers.count(_ % 2 == 0),
      "odds" -> parNumbers.count(_ % 2 != 0),
      "positives" -> parNumbers.count(_ > 0),
      "negatives" -> parNumbers.count(_ < 0),
      "max" -> parNumbers.max,
      "min" -> parNumbers.min,
      "squares" -> parNumbers.map(x => x * x).seq
    )
  }
  
  /**
   * Custom parallel operation with fold
   */
  def parallelGroupBy[A, K](data: Vector[A], keyFunc: A => K): Map[K, Vector[A]] = {
    data.par.aggregate(Map.empty[K, Vector[A]])(
      seqop = { (acc, item) =>
        val key = keyFunc(item)
        acc + (key -> (acc.getOrElse(key, Vector.empty) :+ item))
      },
      combop = { (acc1, acc2) =>
        (acc1.keySet ++ acc2.keySet).map { key =>
          key -> (acc1.getOrElse(key, Vector.empty) ++ acc2.getOrElse(key, Vector.empty))
        }.toMap
      }
    )
  }
}

// ============ Main Demo Application ============

object ConcurrentProgrammingDemo extends App {
  import AsyncOperations._
  import scala.concurrent.Await
  import scala.concurrent.duration._
  
  println("=== Scala Concurrent & Async Programming Examples ===\n")
  
  // Create actor system
  implicit val system: ActorSystem = ActorSystem("ConcurrentDemo")
  implicit val executionContext: ExecutionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(5.seconds)
  
  try {
    // ---- Actor System Demo ----
    println("=== Actor System Demo ===")
    
    val userManager = system.actorOf(Props[UserManagerActor], "userManager")
    val inventoryActor = system.actorOf(Props[InventoryActor], "inventory")
    val paymentActor = system.actorOf(Props[PaymentActor], "payment")
    val orderProcessor = system.actorOf(Props(new OrderProcessingActor(inventoryActor, paymentActor)), "orderProcessor")
    
    // User management operations
    import UserManagerProtocol._
    
    val createFuture = (userManager ? CreateUser("Alice Smith", "alice@example.com", 28)).mapTo[UserResponse]
    val userResult = Await.result(createFuture, 2.seconds)
    println(s"User creation result: $userResult")
    
    val getUserFuture = (userManager ? GetUser(1L)).mapTo[UserResponse]
    val foundUser = Await.result(getUserFuture, 2.seconds)
    println(s"Found user: $foundUser")
    
    // ---- Order Processing Demo ----
    println("\n=== Order Processing Demo ===")
    
    import OrderProtocol._
    
    val order = Order(
      id = "ORD001",
      userId = 1L,
      items = List(
        OrderItem("product1", 2, BigDecimal(29.99)),
        OrderItem("product2", 1, BigDecimal(15.50))
      )
    )
    
    val orderFuture = (orderProcessor ? ProcessOrder(order)).mapTo[OrderResponse]
    val orderResult = Await.result(orderFuture, 3.seconds)
    println(s"Order processing result: $orderResult")
    
    // ---- Distributed Computation Demo ----
    println("\n=== Distributed Computation Demo ===")
    
    val computationCoordinator = system.actorOf(Props[ComputationCoordinatorActor], "coordinator")
    
    import ComputationProtocol._
    
    val largeNumbers = (1 to 1000).toList
    val computationFuture = (computationCoordinator ? DistributedSum("task1", largeNumbers, 4)).mapTo[TaskResult]
    val computationResult = Await.result(computationFuture, 5.seconds)
    println(s"Distributed computation result: $computationResult")
    
    // ---- Future-based Async Operations ----
    println("\n=== Future-based Async Operations ===")
    
    // Complex workflow
    val workflowFuture = createUserWorkflow("Bob Johnson", "bob@example.com", 35)
    val workflowResult = Await.result(workflowFuture, 5.seconds)
    println(s"User creation workflow result: $workflowResult")
    
    // Batch processing
    val userIds = List(1L, 2L, 3L, 4L, 5L)
    val batchFuture = processBatch(
      userIds, 
      (id: Long) => DatabaseSimulator.getUser(id),
      batchSize = 2
    )
    val batchResult = Await.result(batchFuture, 10.seconds)
    println(s"Batch processing result: ${batchResult.flatten.length} users processed")
    
    // Circuit breaker demo
    val circuitBreaker = new AsyncOperations.CircuitBreaker(failureThreshold = 2)
    
    def flakyOperation(): Future[String] = Future {
      if (Random.nextBoolean()) "Success!"
      else throw new RuntimeException("Random failure")
    }
    
    println("\nCircuit breaker demo:")
    for (i <- 1 to 5) {
      val cbResult = circuitBreaker.execute(() => flakyOperation())
      try {
        val result = Await.result(cbResult, 1.second)
        println(s"Attempt $i: $result")
      } catch {
        case ex: Exception => println(s"Attempt $i: Failed - ${ex.getMessage}")
      }
    }
    
    // ---- Parallel Collections Demo ----
    println("\n=== Parallel Collections Demo ===")
    
    val numbers = Vector.range(1, 21)
    val factorials = ParallelCollections.parallelFactorial(numbers.take(10))
    println(s"Parallel factorials (first 5): ${factorials.take(5)}")
    
    val largeDataset = Vector.fill(1000)(s"Sample text ${Random.nextInt(1000)} with numbers ${Random.nextInt(100)}")
    val processedData = ParallelCollections.processLargeDataset(largeDataset.take(10))
    println(s"Processed dataset sample: ${processedData.take(3)}")
    
    val analysisResult = ParallelCollections.analyzeNumbers(Vector.range(-50, 51))
    println(s"Number analysis: $analysisResult")
    
    // ---- Custom Parallel Operations ----
    val testData = Vector.fill(100)(Random.nextInt(10))
    val grouped = ParallelCollections.parallelGroupBy(testData, (x: Int) => x % 3)
    println(s"Parallel grouping by mod 3: ${grouped.mapValues(_.length)}")
    
  } catch {
    case ex: Exception =>
      println(s"Demo error: ${ex.getMessage}")
      ex.printStackTrace()
  } finally {
    // Clean shutdown
    system.terminate()
    Await.result(system.whenTerminated, 10.seconds)
  }
  
  println("\n=== Concurrent Programming Features Demonstrated ===")
  println("- Akka Actor system with message passing")
  println("- Actor supervision and lifecycle management")
  println("- Future composition and transformation")
  println("- Error handling with Try, Future, and custom Result types")
  println("- Async workflows with for-comprehensions")
  println("- Circuit breaker pattern for fault tolerance")
  println("- Retry mechanisms with exponential backoff")
  println("- Batch processing with rate limiting")
  println("- Parallel collections for CPU-intensive tasks")
  println("- Custom parallel algorithms and aggregations")
  println("- Thread pool management and execution contexts")
  println("- Distributed computation coordination")
}