// Sample Scala code for AI training dataset.
// Demonstrates basic algorithms and patterns with functional programming.

import scala.annotation.tailrec
import scala.collection.mutable

/**
 * Collection of basic algorithms for AI training.
 */
object BasicAlgorithms {
  
  /**
   * Implementation of bubble sort algorithm.
   * Time complexity: O(n^2)
   * Space complexity: O(1)
   */
  def bubbleSort[T](arr: Array[T])(implicit ord: Ordering[T]): Unit = {
    val n = arr.length
    for (i <- 0 until n) {
      for (j <- 0 until n - i - 1) {
        if (ord.gt(arr(j), arr(j + 1))) {
          val temp = arr(j)
          arr(j) = arr(j + 1)
          arr(j + 1) = temp
        }
      }
    }
  }
  
  /**
   * Functional bubble sort that returns a new sorted list.
   */
  def bubbleSortFunctional[T: Ordering](list: List[T]): List[T] = {
    def bubblePass(xs: List[T]): List[T] = xs match {
      case x :: y :: rest if Ordering[T].gt(x, y) => y :: bubblePass(x :: rest)
      case x :: rest => x :: bubblePass(rest)
      case Nil => Nil
    }
    
    def sort(xs: List[T]): List[T] = {
      val sorted = bubblePass(xs)
      if (sorted == xs) xs else sort(sorted)
    }
    
    sort(list)
  }
  
  /**
   * Binary search implementation for sorted arrays.
   * Time complexity: O(log n)
   * Space complexity: O(1)
   */
  def binarySearch[T](arr: Array[T], target: T)(implicit ord: Ordering[T]): Option[Int] = {
    @tailrec
    def search(left: Int, right: Int): Option[Int] = {
      if (left > right) None
      else {
        val mid = left + (right - left) / 2
        ord.compare(arr(mid), target) match {
          case 0 => Some(mid)
          case c if c < 0 => search(mid + 1, right)
          case _ => search(left, mid - 1)
        }
      }
    }
    
    search(0, arr.length - 1)
  }
  
  /**
   * Functional binary search for lists.
   */
  def binarySearchList[T: Ordering](list: List[T], target: T): Option[Int] = {
    val arr = list.toArray
    binarySearch(arr, target)
  }
  
  /**
   * Quick sort implementation using functional programming.
   * Time complexity: O(n log n) average, O(n^2) worst case
   * Space complexity: O(log n)
   */
  def quickSort[T: Ordering](list: List[T]): List[T] = list match {
    case Nil => Nil
    case head :: tail =>
      val (left, right) = tail.partition(Ordering[T].lt(_, head))
      quickSort(left) ::: head :: quickSort(right)
  }
  
  /**
   * Merge sort implementation.
   * Time complexity: O(n log n)
   * Space complexity: O(n)
   */
  def mergeSort[T: Ordering](list: List[T]): List[T] = {
    def merge(left: List[T], right: List[T]): List[T] = (left, right) match {
      case (Nil, _) => right
      case (_, Nil) => left
      case (x :: xs, y :: ys) =>
        if (Ordering[T].lteq(x, y)) x :: merge(xs, right)
        else y :: merge(left, ys)
    }
    
    list match {
      case Nil | _ :: Nil => list
      case _ =>
        val (left, right) = list.splitAt(list.length / 2)
        merge(mergeSort(left), mergeSort(right))
    }
  }
  
  /**
   * Fibonacci sequence implementation using recursion with memoization.
   * Time complexity: O(n)
   * Space complexity: O(n)
   */
  lazy val fibonacci: LazyList[BigInt] = {
    def fib(a: BigInt, b: BigInt): LazyList[BigInt] = a #:: fib(b, a + b)
    fib(0, 1)
  }
  
  def fibonacciN(n: Int): BigInt = fibonacci(n)
  
  /**
   * Fibonacci with explicit memoization.
   */
  def fibonacciMemo(n: Int, memo: mutable.Map[Int, BigInt] = mutable.Map()): BigInt = {
    if (n <= 1) n
    else memo.getOrElseUpdate(n, fibonacciMemo(n - 1, memo) + fibonacciMemo(n - 2, memo))
  }
  
  /**
   * Greatest Common Divisor using Euclidean algorithm.
   * Time complexity: O(log min(a, b))
   * Space complexity: O(1)
   */
  @tailrec
  def gcd(a: Int, b: Int): Int = {
    if (b == 0) a else gcd(b, a % b)
  }
  
  /**
   * Least Common Multiple.
   */
  def lcm(a: Int, b: Int): Int = (a * b) / gcd(a, b)
  
  /**
   * Factorial implementation using functional programming.
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  def factorial(n: Int): BigInt = {
    require(n >= 0, "Factorial is not defined for negative numbers")
    (1 to n).foldLeft(BigInt(1))(_ * _)
  }
  
  /**
   * Factorial using tail recursion.
   */
  def factorialTailRec(n: Int): BigInt = {
    @tailrec
    def fact(n: Int, acc: BigInt): BigInt = {
      if (n <= 1) acc else fact(n - 1, acc * n)
    }
    fact(n, 1)
  }
  
  /**
   * Check if a number is prime.
   * Time complexity: O(sqrt(n))
   * Space complexity: O(1)
   */
  def isPrime(n: Int): Boolean = {
    if (n < 2) false
    else if (n == 2) true
    else if (n % 2 == 0) false
    else !(3 to math.sqrt(n).toInt by 2).exists(n % _ == 0)
  }
  
  /**
   * Sieve of Eratosthenes using functional programming.
   * Time complexity: O(n log log n)
   * Space complexity: O(n)
   */
  def sieveOfEratosthenes(n: Int): List[Int] = {
    def sieve(nums: List[Int]): List[Int] = nums match {
      case Nil => Nil
      case head :: tail => head :: sieve(tail.filterNot(_ % head == 0))
    }
    
    if (n < 2) Nil else sieve((2 to n).toList)
  }
  
  /**
   * Two Sum problem using functional approach.
   * Time complexity: O(n)
   * Space complexity: O(n)
   */
  def twoSum(nums: Array[Int], target: Int): Option[(Int, Int)] = {
    val numMap = mutable.Map[Int, Int]()
    
    nums.zipWithIndex.find { case (num, i) =>
      val complement = target - num
      numMap.get(complement) match {
        case Some(j) => true
        case None =>
          numMap(num) = i
          false
      }
    }.flatMap { case (num, i) =>
      numMap.get(target - num).map((_, i))
    }
  }
  
  /**
   * Maximum subarray sum (Kadane's algorithm) using functional programming.
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  def maxSubarraySum(nums: Array[Int]): Int = {
    if (nums.isEmpty) 0
    else {
      nums.tail.foldLeft((nums.head, nums.head)) { case ((maxSum, currentSum), num) =>
        val newCurrentSum = math.max(num, currentSum + num)
        (math.max(maxSum, newCurrentSum), newCurrentSum)
      }._1
    }
  }
  
  /**
   * Valid parentheses checker using pattern matching.
   * Time complexity: O(n)
   * Space complexity: O(n)
   */
  def isValidParentheses(s: String): Boolean = {
    @tailrec
    def check(chars: List[Char], stack: List[Char]): Boolean = chars match {
      case Nil => stack.isEmpty
      case '(' :: rest => check(rest, '(' :: stack)
      case '{' :: rest => check(rest, '{' :: stack)
      case '[' :: rest => check(rest, '[' :: stack)
      case ')' :: rest => stack.headOption.contains('(') && check(rest, stack.tail)
      case '}' :: rest => stack.headOption.contains('{') && check(rest, stack.tail)
      case ']' :: rest => stack.headOption.contains('[') && check(rest, stack.tail)
      case _ :: rest => check(rest, stack)
    }
    
    check(s.toList, Nil)
  }
}

// Data Structures using Scala features

/**
 * Immutable Stack implementation.
 */
sealed trait Stack[+T] {
  def push[U >: T](item: U): Stack[U]
  def pop: (Option[T], Stack[T])
  def peek: Option[T]
  def isEmpty: Boolean
}

case object EmptyStack extends Stack[Nothing] {
  def push[U](item: U): Stack[U] = NonEmptyStack(item, EmptyStack)
  def pop: (Option[Nothing], Stack[Nothing]) = (None, EmptyStack)
  def peek: Option[Nothing] = None
  def isEmpty: Boolean = true
}

case class NonEmptyStack[T](head: T, tail: Stack[T]) extends Stack[T] {
  def push[U >: T](item: U): Stack[U] = NonEmptyStack(item, this)
  def pop: (Option[T], Stack[T]) = (Some(head), tail)
  def peek: Option[T] = Some(head)
  def isEmpty: Boolean = false
}

/**
 * Binary Tree implementation using case classes.
 */
sealed trait Tree[+T]
case object Leaf extends Tree[Nothing]
case class Node[T](value: T, left: Tree[T], right: Tree[T]) extends Tree[T]

object Tree {
  def insert[T: Ordering](tree: Tree[T], value: T): Tree[T] = tree match {
    case Leaf => Node(value, Leaf, Leaf)
    case Node(v, left, right) =>
      if (Ordering[T].lt(value, v)) Node(v, insert(left, value), right)
      else if (Ordering[T].gt(value, v)) Node(v, left, insert(right, value))
      else tree
  }
  
  def search[T: Ordering](tree: Tree[T], value: T): Boolean = tree match {
    case Leaf => false
    case Node(v, left, right) =>
      Ordering[T].compare(value, v) match {
        case 0 => true
        case c if c < 0 => search(left, value)
        case _ => search(right, value)
      }
  }
  
  def inorderTraversal[T](tree: Tree[T]): List[T] = tree match {
    case Leaf => Nil
    case Node(value, left, right) =>
      inorderTraversal(left) ::: value :: inorderTraversal(right)
  }
}

// Design Patterns in Scala

/**
 * Singleton pattern using object.
 */
object DatabaseConnection {
  private val connectionString = "database://localhost:5432"
  
  def executeQuery(query: String): String = {
    s"Executing '$query' on $connectionString"
  }
}

/**
 * Builder pattern using case class and copy method.
 */
case class Computer(
  cpu: String = "",
  ram: Int = 0,
  storage: Int = 0,
  gpu: Option[String] = None,
  bluetooth: Boolean = false,
  wifi: Boolean = false
) {
  require(cpu.nonEmpty, "CPU is required")
  require(ram > 0, "RAM is required")
  require(storage > 0, "Storage is required")
}

object Computer {
  def builder: Computer = Computer(cpu = "default", ram = 1, storage = 1)
}

/**
 * Observer pattern using traits.
 */
trait Observer[T] {
  def update(data: T): Unit
}

trait Subject[T] {
  private var observers: List[Observer[T]] = Nil
  
  def attach(observer: Observer[T]): Unit = {
    observers = observer :: observers
  }
  
  def detach(observer: Observer[T]): Unit = {
    observers = observers.filterNot(_ == observer)
  }
  
  def notify(data: T): Unit = {
    observers.foreach(_.update(data))
  }
}

class EmailNotifier(email: String) extends Observer[String] {
  def update(data: String): Unit = {
    println(s"Email notification to $email: $data")
  }
}

class NotificationSubject extends Subject[String] {
  private var state: String = ""
  
  def setState(newState: String): Unit = {
    state = newState
    notify(state)
  }
  
  def getState: String = state
}

/**
 * Strategy pattern using higher-order functions.
 */
trait SortStrategy[T] {
  def sort(data: List[T]): List[T]
}

class SortContext[T: Ordering] {
  def executeSort(data: List[T], strategy: List[T] => List[T]): List[T] = {
    strategy(data)
  }
}

// Functional Programming Patterns

/**
 * Monadic operations and for-comprehensions.
 */
object FunctionalPatterns {
  
  // Option monad examples
  def safeDivide(x: Double, y: Double): Option[Double] = {
    if (y != 0) Some(x / y) else None
  }
  
  def calculate(a: Double, b: Double, c: Double): Option[Double] = {
    for {
      result1 <- safeDivide(a, b)
      result2 <- safeDivide(result1, c)
    } yield result2
  }
  
  // List monad and flatMap
  def cartesianProduct[A, B](xs: List[A], ys: List[B]): List[(A, B)] = {
    for {
      x <- xs
      y <- ys
    } yield (x, y)
  }
  
  // Higher-order functions
  def compose[A, B, C](f: B => C, g: A => B): A => C = {
    a => f(g(a))
  }
  
  def curry[A, B, C](f: (A, B) => C): A => B => C = {
    a => b => f(a, b)
  }
  
  def uncurry[A, B, C](f: A => B => C): (A, B) => C = {
    (a, b) => f(a)(b)
  }
}

// Implicits and Type Classes

/**
 * Type class pattern example.
 */
trait Show[T] {
  def show(value: T): String
}

object Show {
  implicit val showInt: Show[Int] = (value: Int) => value.toString
  implicit val showString: Show[String] = (value: String) => s"\"$value\""
  implicit def showList[T: Show]: Show[List[T]] = { list =>
    list.map(implicitly[Show[T]].show).mkString("[", ", ", "]")
  }
  
  def apply[T: Show]: Show[T] = implicitly[Show[T]]
}

def printShow[T: Show](value: T): Unit = {
  println(Show[T].show(value))
}

// Example usage and testing
object Main extends App {
  println("=== Scala Algorithm Tests ===")
  
  // Test functional bubble sort
  val arr = List(64, 34, 25, 12, 22, 11, 90)
  println(s"Original list: $arr")
  val sorted = BasicAlgorithms.bubbleSortFunctional(arr)
  println(s"Bubble sorted: $sorted")
  
  // Test binary search
  val sortedArr = List(1, 3, 5, 7, 9, 11, 13)
  val target = 7
  val index = BasicAlgorithms.binarySearchList(sortedArr, target)
  println(s"Binary search for $target: index $index")
  
  // Test Fibonacci
  val n = 10
  println(s"Fibonacci($n) = ${BasicAlgorithms.fibonacciN(n)}")
  println(s"Fibonacci with memo($n) = ${BasicAlgorithms.fibonacciMemo(n)}")
  
  // Test prime checking
  val num = 17
  println(s"Is $num prime? ${BasicAlgorithms.isPrime(num)}")
  
  // Test Sieve of Eratosthenes
  val primes = BasicAlgorithms.sieveOfEratosthenes(30)
  println(s"Primes up to 30: $primes")
  
  // Test data structures
  println("\n=== Data Structure Tests ===")
  
  // Test immutable Stack
  val stack = EmptyStack.push(1).push(2).push(3)
  println(s"Stack peek: ${stack.peek}")
  val (popped, newStack) = stack.pop
  println(s"Stack pop: $popped")
  
  // Test Binary Tree
  val tree = Tree.insert(Tree.insert(Tree.insert(Leaf, 5), 3), 7)
  println(s"Tree inorder: ${Tree.inorderTraversal(tree)}")
  println(s"Tree search 5: ${Tree.search(tree, 5)}")
  
  // Test design patterns
  println("\n=== Design Pattern Tests ===")
  
  // Test Singleton
  val db = DatabaseConnection
  println(db.executeQuery("SELECT * FROM users"))
  
  // Test Builder with copy
  val computer = Computer.builder.copy(
    cpu = "Intel i7",
    ram = 16,
    storage = 512,
    gpu = Some("NVIDIA RTX 4080"),
    wifi = true,
    bluetooth = true
  )
  println(s"Built computer: $computer")
  
  // Test Observer
  val subject = new NotificationSubject
  val emailObserver = new EmailNotifier("user@example.com")
  subject.attach(emailObserver)
  subject.setState("System update available")
  
  // Test functional patterns
  println("\n=== Functional Pattern Tests ===")
  
  // Test for-comprehension
  val calculation = FunctionalPatterns.calculate(10.0, 2.0, 5.0)
  println(s"Calculation result: $calculation")
  
  // Test cartesian product
  val product = FunctionalPatterns.cartesianProduct(List(1, 2), List('a', 'b'))
  println(s"Cartesian product: $product")
  
  // Test type classes
  printShow(42)
  printShow("Hello")
  printShow(List(1, 2, 3))
}