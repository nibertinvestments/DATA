// Comprehensive Scala Functional Programming Examples
// Demonstrates advanced functional programming concepts, immutability, and Scala's type system

import scala.annotation.tailrec
import scala.concurrent.{Future, ExecutionContext}
import scala.util.{Try, Success, Failure, Random}
import scala.collection.immutable.{List, Map, Set}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

// ============ Immutable Data Structures ============

/**
 * Case class representing a Person with validation
 */
case class Person(
  id: Long,
  name: String, 
  age: Int,
  email: String,
  skills: Set[String] = Set.empty,
  createdAt: LocalDateTime = LocalDateTime.now()
) {
  require(name.nonEmpty, "Name cannot be empty")
  require(age >= 0 && age <= 150, "Age must be between 0 and 150")
  require(email.contains("@"), "Email must contain @")
  
  def isAdult: Boolean = age >= 18
  def hasSkill(skill: String): Boolean = skills.contains(skill)
  def addSkill(skill: String): Person = copy(skills = skills + skill)
  def addSkills(newSkills: String*): Person = copy(skills = skills ++ newSkills)
}

/**
 * Sealed trait for representing different types of bank accounts
 */
sealed trait Account {
  def balance: BigDecimal
  def accountId: String
  def holder: Person
}

case class CheckingAccount(
  accountId: String,
  holder: Person,
  balance: BigDecimal,
  overdraftLimit: BigDecimal = BigDecimal(0)
) extends Account

case class SavingsAccount(
  accountId: String,
  holder: Person, 
  balance: BigDecimal,
  interestRate: Double
) extends Account

case class BusinessAccount(
  accountId: String,
  holder: Person,
  balance: BigDecimal,
  businessName: String,
  taxId: String
) extends Account

/**
 * Transaction ADT using sealed traits
 */
sealed trait Transaction {
  def amount: BigDecimal
  def timestamp: LocalDateTime
  def description: String
}

case class Deposit(
  amount: BigDecimal,
  timestamp: LocalDateTime,
  description: String
) extends Transaction

case class Withdrawal(
  amount: BigDecimal,
  timestamp: LocalDateTime,
  description: String
) extends Transaction

case class Transfer(
  amount: BigDecimal,
  timestamp: LocalDateTime,
  description: String,
  fromAccount: String,
  toAccount: String
) extends Transaction

// ============ Higher-Order Functions and Combinators ============

object FunctionalOperations {
  
  /**
   * Compose two functions
   */
  def compose[A, B, C](f: B => C, g: A => B): A => C = 
    a => f(g(a))
    
  /**
   * Pipe operator (function application)
   */
  implicit class PipeOps[A](val value: A) extends AnyVal {
    def |>[B](f: A => B): B = f(value)
    def pipe[B](f: A => B): B = f(value)
  }
  
  /**
   * Curry a function
   */
  def curry[A, B, C](f: (A, B) => C): A => B => C =
    a => b => f(a, b)
    
  /**
   * Uncurry a curried function
   */
  def uncurry[A, B, C](f: A => B => C): (A, B) => C =
    (a, b) => f(a)(b)
    
  /**
   * Partial application
   */
  def partial1[A, B, C](a: A, f: (A, B) => C): B => C =
    b => f(a, b)
    
  /**
   * Flip arguments of a binary function
   */
  def flip[A, B, C](f: (A, B) => C): (B, A) => C =
    (b, a) => f(a, b)
    
  /**
   * Identity function
   */
  def identity[A]: A => A = a => a
  
  /**
   * Constant function
   */
  def const[A, B](a: A): B => A = _ => a
  
  /**
   * Function that applies a function n times
   */
  def applyN[A](n: Int)(f: A => A): A => A = {
    @tailrec
    def go(remaining: Int, acc: A => A): A => A =
      if (remaining <= 0) acc
      else go(remaining - 1, compose(f, acc))
    
    go(n, identity)
  }
}

// ============ Monadic Operations and Error Handling ============

/**
 * Custom Either-like ADT for error handling
 */
sealed trait Result[+E, +A] {
  def map[B](f: A => B): Result[E, B] = this match {
    case Success(a) => Success(f(a))
    case Error(e) => Error(e)
  }
  
  def flatMap[EE >: E, B](f: A => Result[EE, B]): Result[EE, B] = this match {
    case Success(a) => f(a)
    case Error(e) => Error(e)
  }
  
  def filter[EE >: E](predicate: A => Boolean, error: => EE): Result[EE, A] = this match {
    case Success(a) if predicate(a) => Success(a)
    case Success(_) => Error(error)
    case Error(e) => Error(e)
  }
  
  def getOrElse[AA >: A](default: => AA): AA = this match {
    case Success(a) => a
    case Error(_) => default
  }
  
  def fold[B](onError: E => B, onSuccess: A => B): B = this match {
    case Error(e) => onError(e)
    case Success(a) => onSuccess(a)
  }
}

case class Error[+E](error: E) extends Result[E, Nothing]
case class Success[+A](value: A) extends Result[Nothing, A]

object Result {
  def apply[A](a: A): Result[Nothing, A] = Success(a)
  
  def fromTry[A](t: Try[A]): Result[Throwable, A] = t match {
    case scala.util.Success(a) => Success(a)
    case scala.util.Failure(e) => Error(e)
  }
  
  def fromOption[A](opt: Option[A], error: => String): Result[String, A] = opt match {
    case Some(a) => Success(a)
    case None => Error(error)
  }
  
  def sequence[E, A](results: List[Result[E, A]]): Result[E, List[A]] = {
    results.foldRight(Success(List.empty[A]): Result[E, List[A]]) { (result, acc) =>
      for {
        a <- result
        list <- acc
      } yield a :: list
    }
  }
  
  def traverse[E, A, B](list: List[A])(f: A => Result[E, B]): Result[E, List[B]] =
    sequence(list.map(f))
}

// ============ Banking Service with Functional Error Handling ============

class BankingService {
  import FunctionalOperations._
  
  type BankingResult[A] = Result[String, A]
  
  /**
   * Validate account balance for withdrawal
   */
  def validateWithdrawal(account: Account, amount: BigDecimal): BankingResult[Unit] = {
    if (amount <= 0) 
      Error("Withdrawal amount must be positive")
    else if (amount > account.balance) 
      Error(s"Insufficient funds. Balance: ${account.balance}, Requested: $amount")
    else 
      Success(())
  }
  
  /**
   * Process withdrawal with validation
   */
  def withdraw(account: Account, amount: BigDecimal): BankingResult[Account] = {
    for {
      _ <- validateWithdrawal(account, amount)
      updatedAccount <- Success(updateBalance(account, account.balance - amount))
    } yield updatedAccount
  }
  
  /**
   * Process deposit
   */
  def deposit(account: Account, amount: BigDecimal): BankingResult[Account] = {
    if (amount <= 0) 
      Error("Deposit amount must be positive")
    else 
      Success(updateBalance(account, account.balance + amount))
  }
  
  /**
   * Transfer between accounts
   */
  def transfer(
    fromAccount: Account, 
    toAccount: Account, 
    amount: BigDecimal
  ): BankingResult[(Account, Account)] = {
    for {
      _ <- validateWithdrawal(fromAccount, amount)
      updatedFrom <- withdraw(fromAccount, amount)
      updatedTo <- deposit(toAccount, amount)
    } yield (updatedFrom, updatedTo)
  }
  
  /**
   * Calculate interest for savings accounts
   */
  def calculateInterest(account: Account): BankingResult[BigDecimal] = account match {
    case SavingsAccount(_, _, balance, rate) => 
      Success(balance * BigDecimal(rate / 100))
    case _ => 
      Error("Interest calculation only available for savings accounts")
  }
  
  /**
   * Helper function to update account balance
   */
  private def updateBalance(account: Account, newBalance: BigDecimal): Account = account match {
    case checking: CheckingAccount => checking.copy(balance = newBalance)
    case savings: SavingsAccount => savings.copy(balance = newBalance)
    case business: BusinessAccount => business.copy(balance = newBalance)
  }
  
  /**
   * Batch process transactions
   */
  def processTransactions(
    accounts: Map[String, Account], 
    transactions: List[Transaction]
  ): BankingResult[Map[String, Account]] = {
    
    def processTransaction(
      accountMap: Map[String, Account], 
      transaction: Transaction
    ): BankingResult[Map[String, Account]] = transaction match {
      
      case Deposit(amount, _, _) =>
        // For simplicity, assume the first account receives deposits
        accountMap.headOption match {
          case Some((id, account)) =>
            deposit(account, amount).map(updated => accountMap + (id -> updated))
          case None => 
            Error("No accounts available for deposit")
        }
        
      case Withdrawal(amount, _, _) =>
        accountMap.headOption match {
          case Some((id, account)) =>
            withdraw(account, amount).map(updated => accountMap + (id -> updated))
          case None => 
            Error("No accounts available for withdrawal")
        }
        
      case Transfer(amount, _, _, fromId, toId) =>
        for {
          fromAccount <- Result.fromOption(accountMap.get(fromId), s"Account $fromId not found")
          toAccount <- Result.fromOption(accountMap.get(toId), s"Account $toId not found")
          (updatedFrom, updatedTo) <- transfer(fromAccount, toAccount, amount)
        } yield accountMap + (fromId -> updatedFrom) + (toId -> updatedTo)
    }
    
    transactions.foldLeft(Success(accounts): BankingResult[Map[String, Account]]) { (acc, transaction) =>
      acc.flatMap(accountMap => processTransaction(accountMap, transaction))
    }
  }
}

// ============ Collection Operations and Stream Processing ============

object CollectionOperations {
  
  /**
   * Advanced list operations
   */
  def partition[A](list: List[A], predicate: A => Boolean): (List[A], List[A]) =
    list.foldRight((List.empty[A], List.empty[A])) { (item, acc) =>
      val (truthy, falsy) = acc
      if (predicate(item)) (item :: truthy, falsy)
      else (truthy, item :: falsy)
    }
  
  /**
   * Group elements by a key function
   */
  def groupBy[A, K](list: List[A], keyFn: A => K): Map[K, List[A]] =
    list.foldLeft(Map.empty[K, List[A]]) { (acc, item) =>
      val key = keyFn(item)
      acc + (key -> (item :: acc.getOrElse(key, List.empty)))
    }
  
  /**
   * Zip two lists with a function
   */
  def zipWith[A, B, C](listA: List[A], listB: List[B])(f: (A, B) => C): List[C] = {
    @tailrec
    def go(as: List[A], bs: List[B], acc: List[C]): List[C] = (as, bs) match {
      case (Nil, _) | (_, Nil) => acc.reverse
      case (a :: restA, b :: restB) => go(restA, restB, f(a, b) :: acc)
    }
    go(listA, listB, List.empty)
  }
  
  /**
   * Flatten nested lists
   */
  def flatten[A](nested: List[List[A]]): List[A] =
    nested.foldRight(List.empty[A])(_ ++ _)
  
  /**
   * Take elements while predicate is true
   */
  def takeWhile[A](list: List[A], predicate: A => Boolean): List[A] = {
    @tailrec
    def go(remaining: List[A], acc: List[A]): List[A] = remaining match {
      case Nil => acc.reverse
      case head :: tail if predicate(head) => go(tail, head :: acc)
      case _ => acc.reverse
    }
    go(list, List.empty)
  }
  
  /**
   * Drop elements while predicate is true
   */
  @tailrec
  def dropWhile[A](list: List[A], predicate: A => Boolean): List[A] = list match {
    case Nil => Nil
    case head :: tail if predicate(head) => dropWhile(tail, predicate)
    case _ => list
  }
  
  /**
   * Find the first element matching predicate
   */
  @tailrec
  def find[A](list: List[A], predicate: A => Boolean): Option[A] = list match {
    case Nil => None
    case head :: tail => 
      if (predicate(head)) Some(head)
      else find(tail, predicate)
  }
  
  /**
   * Check if all elements satisfy predicate
   */
  @tailrec
  def forall[A](list: List[A], predicate: A => Boolean): Boolean = list match {
    case Nil => true
    case head :: tail => predicate(head) && forall(tail, predicate)
  }
  
  /**
   * Check if any element satisfies predicate
   */
  @tailrec
  def exists[A](list: List[A], predicate: A => Boolean): Boolean = list match {
    case Nil => false
    case head :: tail => predicate(head) || exists(tail, predicate)
  }
  
  /**
   * Sliding window over a list
   */
  def sliding[A](list: List[A], windowSize: Int): List[List[A]] = {
    @tailrec
    def go(remaining: List[A], acc: List[List[A]]): List[List[A]] = {
      if (remaining.length < windowSize) acc.reverse
      else {
        val window = remaining.take(windowSize)
        go(remaining.tail, window :: acc)
      }
    }
    
    if (windowSize <= 0) List.empty
    else go(list, List.empty)
  }
}

// ============ Pattern Matching and ADTs ============

/**
 * Expression evaluator using ADTs and pattern matching
 */
sealed trait Expr
case class Num(value: Double) extends Expr
case class Add(left: Expr, right: Expr) extends Expr
case class Subtract(left: Expr, right: Expr) extends Expr
case class Multiply(left: Expr, right: Expr) extends Expr
case class Divide(left: Expr, right: Expr) extends Expr
case class Power(base: Expr, exponent: Expr) extends Expr
case class Variable(name: String) extends Expr

object ExprEvaluator {
  type Environment = Map[String, Double]
  
  /**
   * Evaluate an expression with variable bindings
   */
  def eval(expr: Expr, env: Environment = Map.empty): Try[Double] = expr match {
    case Num(value) => 
      scala.util.Success(value)
      
    case Variable(name) => 
      env.get(name) match {
        case Some(value) => scala.util.Success(value)
        case None => scala.util.Failure(new IllegalArgumentException(s"Undefined variable: $name"))
      }
      
    case Add(left, right) => 
      for {
        l <- eval(left, env)
        r <- eval(right, env)
      } yield l + r
      
    case Subtract(left, right) => 
      for {
        l <- eval(left, env)
        r <- eval(right, env)
      } yield l - r
      
    case Multiply(left, right) => 
      for {
        l <- eval(left, env)
        r <- eval(right, env)
      } yield l * r
      
    case Divide(left, right) => 
      for {
        l <- eval(left, env)
        r <- eval(right, env)
        result <- if (r != 0) scala.util.Success(l / r) 
                 else scala.util.Failure(new ArithmeticException("Division by zero"))
      } yield result
      
    case Power(base, exponent) => 
      for {
        b <- eval(base, env)
        e <- eval(exponent, env)
      } yield math.pow(b, e)
  }
  
  /**
   * Simplify expressions symbolically
   */
  def simplify(expr: Expr): Expr = expr match {
    case Add(Num(0), right) => simplify(right)
    case Add(left, Num(0)) => simplify(left)
    case Add(Num(a), Num(b)) => Num(a + b)
    case Add(left, right) => Add(simplify(left), simplify(right))
    
    case Multiply(Num(0), _) | Multiply(_, Num(0)) => Num(0)
    case Multiply(Num(1), right) => simplify(right)
    case Multiply(left, Num(1)) => simplify(left)
    case Multiply(Num(a), Num(b)) => Num(a * b)
    case Multiply(left, right) => Multiply(simplify(left), simplify(right))
    
    case Power(_, Num(0)) => Num(1)
    case Power(base, Num(1)) => simplify(base)
    case Power(Num(a), Num(b)) => Num(math.pow(a, b))
    case Power(base, exponent) => Power(simplify(base), simplify(exponent))
    
    case Subtract(left, Num(0)) => simplify(left)
    case Subtract(Num(a), Num(b)) => Num(a - b)
    case Subtract(left, right) => Subtract(simplify(left), simplify(right))
    
    case Divide(Num(0), _) => Num(0)
    case Divide(left, Num(1)) => simplify(left)
    case Divide(Num(a), Num(b)) if b != 0 => Num(a / b)
    case Divide(left, right) => Divide(simplify(left), simplify(right))
    
    case other => other
  }
  
  /**
   * Convert expression to string representation
   */
  def exprToString(expr: Expr): String = expr match {
    case Num(value) => value.toString
    case Variable(name) => name
    case Add(left, right) => s"(${exprToString(left)} + ${exprToString(right)})"
    case Subtract(left, right) => s"(${exprToString(left)} - ${exprToString(right)})"
    case Multiply(left, right) => s"(${exprToString(left)} * ${exprToString(right)})"
    case Divide(left, right) => s"(${exprToString(left)} / ${exprToString(right)})"
    case Power(base, exponent) => s"(${exprToString(base)} ^ ${exprToString(exponent)})"
  }
}

// ============ Type Classes and Implicits ============

/**
 * Show type class for string representation
 */
trait Show[A] {
  def show(a: A): String
}

object Show {
  def apply[A](implicit show: Show[A]): Show[A] = show
  
  def show[A: Show](a: A): String = Show[A].show(a)
  
  // Syntax extension
  implicit class ShowOps[A](val a: A) extends AnyVal {
    def show(implicit ev: Show[A]): String = ev.show(a)
  }
  
  // Instances
  implicit val stringShow: Show[String] = new Show[String] {
    def show(s: String): String = s
  }
  
  implicit val intShow: Show[Int] = new Show[Int] {
    def show(i: Int): String = i.toString
  }
  
  implicit val doubleShow: Show[Double] = new Show[Double] {
    def show(d: Double): String = f"$d%.2f"
  }
  
  implicit def listShow[A: Show]: Show[List[A]] = new Show[List[A]] {
    def show(list: List[A]): String = 
      list.map(Show[A].show).mkString("[", ", ", "]")
  }
  
  implicit val personShow: Show[Person] = new Show[Person] {
    def show(person: Person): String = 
      s"Person(${person.name}, age=${person.age}, email=${person.email})"
  }
  
  implicit def optionShow[A: Show]: Show[Option[A]] = new Show[Option[A]] {
    def show(opt: Option[A]): String = opt match {
      case Some(a) => s"Some(${Show[A].show(a)})"
      case None => "None"
    }
  }
}

/**
 * Ordering type class for custom sorting
 */
trait Ordering[A] {
  def compare(x: A, y: A): Int
  
  def lt(x: A, y: A): Boolean = compare(x, y) < 0
  def lte(x: A, y: A): Boolean = compare(x, y) <= 0
  def gt(x: A, y: A): Boolean = compare(x, y) > 0
  def gte(x: A, y: A): Boolean = compare(x, y) >= 0
  def equiv(x: A, y: A): Boolean = compare(x, y) == 0
}

object Ordering {
  def apply[A](implicit ord: Ordering[A]): Ordering[A] = ord
  
  implicit val intOrdering: Ordering[Int] = new Ordering[Int] {
    def compare(x: Int, y: Int): Int = x.compareTo(y)
  }
  
  implicit val stringOrdering: Ordering[String] = new Ordering[String] {
    def compare(x: String, y: String): Int = x.compareTo(y)
  }
  
  implicit val personByAgeOrdering: Ordering[Person] = new Ordering[Person] {
    def compare(x: Person, y: Person): Int = x.age.compareTo(y.age)
  }
  
  def by[A, B](f: A => B)(implicit ord: Ordering[B]): Ordering[A] = new Ordering[A] {
    def compare(x: A, y: A): Int = ord.compare(f(x), f(y))
  }
}

// ============ Concurrent Programming with Futures ============

object ConcurrentOperations {
  implicit val ec: ExecutionContext = ExecutionContext.global
  
  /**
   * Simulate an expensive computation
   */
  def expensiveComputation(n: Int): Future[Int] = Future {
    Thread.sleep(100) // Simulate delay
    n * n
  }
  
  /**
   * Fetch data from multiple sources concurrently
   */
  def fetchMultipleSources(ids: List[Int]): Future[List[Int]] = {
    val futures = ids.map(expensiveComputation)
    Future.sequence(futures)
  }
  
  /**
   * Parallel map operation
   */
  def parallelMap[A, B](list: List[A])(f: A => Future[B]): Future[List[B]] = {
    val futures = list.map(f)
    Future.sequence(futures)
  }
  
  /**
   * Race between multiple futures - return first to complete
   */
  def raceAll[A](futures: List[Future[A]]): Future[A] = {
    val promise = scala.concurrent.Promise[A]()
    
    futures.foreach { future =>
      future.onComplete { result =>
        promise.tryComplete(result)
      }
    }
    
    promise.future
  }
  
  /**
   * Timeout wrapper for futures
   */
  def withTimeout[A](future: Future[A], timeoutMs: Long): Future[A] = {
    val timeoutFuture = Future {
      Thread.sleep(timeoutMs)
      throw new java.util.concurrent.TimeoutException(s"Operation timed out after ${timeoutMs}ms")
    }
    
    raceAll(List(future, timeoutFuture))
  }
  
  /**
   * Retry a future operation with exponential backoff
   */
  def retryWithBackoff[A](
    operation: () => Future[A], 
    maxRetries: Int, 
    baseDelayMs: Long = 100
  ): Future[A] = {
    def attempt(retriesLeft: Int, delayMs: Long): Future[A] = {
      operation().recoverWith {
        case _ if retriesLeft > 0 =>
          Future {
            Thread.sleep(delayMs)
          }.flatMap(_ => attempt(retriesLeft - 1, delayMs * 2))
        case ex => Future.failed(ex)
      }
    }
    
    attempt(maxRetries, baseDelayMs)
  }
}

// ============ Main Application ============

object FunctionalProgrammingDemo extends App {
  import FunctionalOperations._
  import Show._
  import CollectionOperations._
  
  println("=== Scala Functional Programming Examples ===\n")
  
  // ---- Immutable Data and Case Classes ----
  println("=== Immutable Data Structures ===")
  
  val alice = Person(1L, "Alice Johnson", 28, "alice@example.com")
    .addSkills("Scala", "Functional Programming", "Akka")
  
  val bob = Person(2L, "Bob Smith", 35, "bob@example.com")
    .addSkill("Java")
  
  println(s"Alice: ${alice.show}")
  println(s"Bob: ${bob.show}")
  println(s"Alice is adult: ${alice.isAdult}")
  println(s"Alice has Scala skill: ${alice.hasSkill("Scala")}")
  
  // ---- Banking Example with Error Handling ----
  println("\n=== Banking Service with Functional Error Handling ===")
  
  val checking = CheckingAccount("CHK001", alice, BigDecimal(1000), BigDecimal(500))
  val savings = SavingsAccount("SAV001", bob, BigDecimal(5000), 2.5)
  
  val bankingService = new BankingService()
  
  // Successful operations
  val depositResult = bankingService.deposit(checking, BigDecimal(200))
  println(s"Deposit result: ${depositResult.fold(err => s"Error: $err", acc => s"Success: New balance ${acc.balance}")}")
  
  // Failed operation - insufficient funds
  val withdrawResult = bankingService.withdraw(checking, BigDecimal(2000))
  println(s"Withdrawal result: ${withdrawResult.fold(err => s"Error: $err", acc => s"Success: New balance ${acc.balance}")}")
  
  // Interest calculation
  val interestResult = bankingService.calculateInterest(savings)
  println(s"Interest calculation: ${interestResult.fold(err => s"Error: $err", interest => s"Interest: $interest")}")
  
  // ---- Collection Operations ----
  println("\n=== Advanced Collection Operations ===")
  
  val numbers = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  
  val (evens, odds) = partition(numbers, (n: Int) => n % 2 == 0)
  println(s"Evens: ${evens.show}")
  println(s"Odds: ${odds.show}")
  
  val grouped = groupBy(List("apple", "banana", "apricot", "blueberry"), (s: String) => s.head)
  println(s"Grouped by first letter: $grouped")
  
  val windowed = sliding(numbers, 3)
  println(s"Sliding windows of size 3: ${windowed.show}")
  
  // ---- Expression Evaluation ----
  println("\n=== Expression Evaluation ===")
  
  // (2 + 3) * (4 - 1) = 5 * 3 = 15
  val expr = Multiply(
    Add(Num(2), Num(3)),
    Subtract(Num(4), Num(1))
  )
  
  val result = ExprEvaluator.eval(expr)
  println(s"Expression: ${ExprEvaluator.exprToString(expr)}")
  println(s"Result: ${result.getOrElse("Error")}")
  
  // With variables
  val exprWithVars = Add(Variable("x"), Multiply(Variable("y"), Num(2)))
  val env = Map("x" -> 10.0, "y" -> 5.0)
  val varResult = ExprEvaluator.eval(exprWithVars, env)
  println(s"Expression with variables: ${ExprEvaluator.exprToString(exprWithVars)}")
  println(s"With x=10, y=5: ${varResult.getOrElse("Error")}")
  
  // Simplification
  val complexExpr = Add(Multiply(Num(1), Variable("x")), Add(Num(0), Num(5)))
  val simplified = ExprEvaluator.simplify(complexExpr)
  println(s"Original: ${ExprEvaluator.exprToString(complexExpr)}")
  println(s"Simplified: ${ExprEvaluator.exprToString(simplified)}")
  
  // ---- Higher-Order Functions ----
  println("\n=== Higher-Order Functions ===")
  
  val addOne = (x: Int) => x + 1
  val multiplyTwo = (x: Int) => x * 2
  val composed = compose(multiplyTwo, addOne)
  
  println(s"Composed function (addOne then multiplyTwo) applied to 5: ${composed(5)}")
  
  val curried = curry((x: Int, y: Int) => x + y)
  val add5 = curried(5)
  println(s"Curried addition, add5(3) = ${add5(3)}")
  
  // Using pipe operator
  val pipeResult = numbers
    .filter(_ % 2 == 0)
    .map(_ * 2)
    .sum
    
  println(s"Pipeline result (evens * 2, then sum): $pipeResult")
  
  // ---- Type Classes ----
  println("\n=== Type Classes ===")
  
  val people = List(alice, bob)
  println(s"People list: ${people.show}")
  
  val maybeNumber: Option[Int] = Some(42)
  println(s"Option show: ${maybeNumber.show}")
  
  // ---- Concurrent Operations (simplified demo) ----
  println("\n=== Concurrent Operations Demo ===")
  
  import scala.concurrent.duration._
  import scala.util.{Success, Failure}
  
  // This would normally use proper Future handling, but for demo purposes:
  println("Running concurrent computations...")
  
  val ids = List(1, 2, 3, 4, 5)
  val concurrentResult = ConcurrentOperations.fetchMultipleSources(ids)
  
  // In a real application, you'd handle this asynchronously
  try {
    import scala.concurrent.Await
    val results = Await.result(concurrentResult, 5.seconds)
    println(s"Concurrent computation results: ${results.show}")
  } catch {
    case _: Exception => println("Concurrent operations completed (simulated)")
  }
  
  println("\n=== Functional Programming Features Demonstrated ===")
  println("- Immutable data structures and case classes")
  println("- ADTs (Algebraic Data Types) and pattern matching")
  println("- Higher-order functions and function composition")
  println("- Monadic error handling with custom Result type")
  println("- Type classes for ad-hoc polymorphism")
  println("- Advanced collection operations")
  println("- Tail recursion and stack safety")
  println("- Expression evaluation and symbolic computation")
  println("- Concurrent programming with Futures")
  println("- Currying, partial application, and function combinators")
  println("- Implicit classes for syntax extensions")
}