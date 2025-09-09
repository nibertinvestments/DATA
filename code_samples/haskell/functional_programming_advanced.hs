{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}

-- Advanced Haskell Programming Examples
-- This module demonstrates intermediate to advanced Haskell concepts including:
-- - Advanced type system features
-- - Monads and monad transformers
-- - Functional data structures
-- - Parser combinators
-- - Concurrency and parallelism
-- - Category theory applications

module AdvancedHaskell where

import Control.Applicative
import Control.Monad
import Control.Monad.State
import Control.Monad.Reader
import Control.Monad.Writer
import Control.Monad.IO.Class (liftIO)
import Control.Concurrent
import Control.Concurrent.STM
import Control.Parallel.Strategies
import Data.List (foldl', sortBy, groupBy, nub)
import Data.Map (Map)
import qualified Data.Map as M
import Data.Set (Set)
import qualified Data.Set as S
import Data.Time
import Data.Monoid
import Text.Parsec
import Text.Parsec.String

-- | Advanced Data Structures
-- =========================

-- | Binary Tree with various operations
data Tree a = Empty | Node a (Tree a) (Tree a)
    deriving (Show, Eq)

instance Functor Tree where
    fmap _ Empty = Empty
    fmap f (Node x left right) = Node (f x) (fmap f left) (fmap f right)

instance Foldable Tree where
    foldr _ z Empty = z
    foldr f z (Node x left right) = f x (foldr f (foldr f z right) left)

instance Traversable Tree where
    traverse _ Empty = pure Empty
    traverse f (Node x left right) = 
        Node <$> f x <*> traverse f left <*> traverse f right

-- | Insert into binary search tree
insertBST :: Ord a => a -> Tree a -> Tree a
insertBST x Empty = Node x Empty Empty
insertBST x (Node y left right)
    | x <= y    = Node y (insertBST x left) right
    | otherwise = Node y left (insertBST x right)

-- | Search in binary search tree
searchBST :: Ord a => a -> Tree a -> Bool
searchBST _ Empty = False
searchBST x (Node y left right)
    | x == y    = True
    | x < y     = searchBST x left
    | otherwise = searchBST x right

-- | Tree traversal methods
inOrderTraversal :: Tree a -> [a]
inOrderTraversal Empty = []
inOrderTraversal (Node x left right) = 
    inOrderTraversal left ++ [x] ++ inOrderTraversal right

preOrderTraversal :: Tree a -> [a]
preOrderTraversal Empty = []
preOrderTraversal (Node x left right) = 
    [x] ++ preOrderTraversal left ++ preOrderTraversal right

postOrderTraversal :: Tree a -> [a]
postOrderTraversal Empty = []
postOrderTraversal (Node x left right) = 
    postOrderTraversal left ++ postOrderTraversal right ++ [x]

-- | Persistent List with efficient operations
data PersistentList a = Nil | Cons a (PersistentList a)
    deriving (Show, Eq)

instance Functor PersistentList where
    fmap _ Nil = Nil
    fmap f (Cons x xs) = Cons (f x) (fmap f xs)

instance Foldable PersistentList where
    foldr _ z Nil = z
    foldr f z (Cons x xs) = f x (foldr f z xs)

-- | Efficient list concatenation using difference lists
newtype DiffList a = DL ([a] -> [a])

instance Semigroup (DiffList a) where
    DL f <> DL g = DL (f . g)

instance Monoid (DiffList a) where
    mempty = DL id

fromList :: [a] -> DiffList a
fromList xs = DL (xs ++)

toList :: DiffList a -> [a]
toList (DL f) = f []

-- | Monadic Programming Patterns
-- =============================

-- | Application configuration with Reader monad
data AppConfig = AppConfig
    { dbConnection :: String
    , apiKey       :: String
    , debugMode    :: Bool
    } deriving (Show)

-- | Application monad stack
type App = ReaderT AppConfig (StateT AppState IO)

data AppState = AppState
    { requestCount :: Int
    , errorCount   :: Int
    , lastRequest  :: UTCTime
    } deriving (Show)

-- | Run application with configuration
runApp :: AppConfig -> AppState -> App a -> IO (a, AppState)
runApp config state action = runStateT (runReaderT action config) state

-- | Example application functions
logRequest :: String -> App ()
logRequest msg = do
    config <- ask
    state <- get
    currentTime <- liftIO getCurrentTime
    
    put state { requestCount = requestCount state + 1
              , lastRequest = currentTime }
    
    when (debugMode config) $
        liftIO $ putStrLn $ "LOG: " ++ msg

handleError :: String -> App ()
handleError err = do
    modify $ \s -> s { errorCount = errorCount s + 1 }
    liftIO $ putStrLn $ "ERROR: " ++ err

-- | Database operations simulation
fetchUser :: Int -> App (Maybe String)
fetchUser userId = do
    config <- ask
    logRequest $ "Fetching user " ++ show userId
    
    -- Simulate database operation
    liftIO $ do
        putStrLn $ "Connecting to: " ++ dbConnection config
        threadDelay 100000  -- Simulate network delay
        return $ Just $ "User" ++ show userId

-- | Advanced Type System Features
-- ==============================

-- | Multi-parameter type class with functional dependencies
class Collection c e | c -> e where
    empty :: c
    insert :: e -> c -> c
    member :: e -> c -> Bool
    size :: c -> Int

-- | List instance
instance Eq a => Collection [a] a where
    empty = []
    insert = (:)
    member = elem
    size = length

-- | Set instance
instance Ord a => Collection (Set a) a where
    empty = S.empty
    insert = S.insert
    member = S.member
    size = S.size

-- | Generalized algebraic data types (GADTs)
data Expr a where
    IntLit  :: Int -> Expr Int
    BoolLit :: Bool -> Expr Bool
    Add     :: Expr Int -> Expr Int -> Expr Int
    Mul     :: Expr Int -> Expr Int -> Expr Int
    Eq      :: Expr Int -> Expr Int -> Expr Bool
    If      :: Expr Bool -> Expr a -> Expr a -> Expr a

-- | Type-safe expression evaluator
eval :: Expr a -> a
eval (IntLit n) = n
eval (BoolLit b) = b
eval (Add e1 e2) = eval e1 + eval e2
eval (Mul e1 e2) = eval e1 * eval e2
eval (Eq e1 e2) = eval e1 == eval e2
eval (If cond t f) = if eval cond then eval t else eval f

-- | Parser Combinators
-- ===================

-- | Simple arithmetic expression parser
data ArithExpr = 
    Num Int |
    Plus ArithExpr ArithExpr |
    Minus ArithExpr ArithExpr |
    Times ArithExpr ArithExpr |
    Divide ArithExpr ArithExpr
    deriving (Show, Eq)

-- | Parse a number
parseNumber :: Parser ArithExpr
parseNumber = do
    n <- many1 digit
    return $ Num (read n)

-- | Parse an expression with precedence
parseExpression :: Parser ArithExpr
parseExpression = buildExpressionParser operatorTable parseTerm

operatorTable = [
    [Infix (char '*' >> return Times) AssocLeft,
     Infix (char '/' >> return Divide) AssocLeft],
    [Infix (char '+' >> return Plus) AssocLeft,
     Infix (char '-' >> return Minus) AssocLeft]
    ]

parseTerm :: Parser ArithExpr
parseTerm = parseNumber <|> between (char '(') (char ')') parseExpression

-- | Evaluate arithmetic expressions
evalArith :: ArithExpr -> Maybe Int
evalArith (Num n) = Just n
evalArith (Plus e1 e2) = liftA2 (+) (evalArith e1) (evalArith e2)
evalArith (Minus e1 e2) = liftA2 (-) (evalArith e1) (evalArith e2)
evalArith (Times e1 e2) = liftA2 (*) (evalArith e1) (evalArith e2)
evalArith (Divide e1 e2) = do
    v1 <- evalArith e1
    v2 <- evalArith e2
    if v2 == 0 then Nothing else Just (v1 `div` v2)

-- | Concurrency and Parallelism
-- =============================

-- | Software Transactional Memory example
type Account = TVar Int

transfer :: Account -> Account -> Int -> STM ()
transfer from to amount = do
    fromBalance <- readTVar from
    toBalance <- readTVar to
    if fromBalance >= amount
        then do
            writeTVar from (fromBalance - amount)
            writeTVar to (toBalance + amount)
        else retry

-- | Parallel computation strategies
parallelMap :: (a -> b) -> [a] -> [b]
parallelMap f xs = map f xs `using` parList rdeepseq

-- | Parallel fold with strategies
parallelSum :: [Int] -> Int
parallelSum xs = sum xs `using` rdeepseq

-- | Fork multiple threads for concurrent processing
concurrentProcessing :: [String] -> IO [String]
concurrentProcessing items = do
    mvars <- mapM (\item -> do
        mvar <- newEmptyMVar
        forkIO $ do
            result <- processItem item
            putMVar mvar result
        return mvar) items
    
    mapM takeMVar mvars
  where
    processItem :: String -> IO String
    processItem item = do
        threadDelay 1000000  -- Simulate work
        return $ "Processed: " ++ item

-- | Higher-Order Functions and Combinators
-- ========================================

-- | Function composition combinator
(.:) :: (c -> d) -> (a -> b -> c) -> a -> b -> d
(.:) = (.) . (.)

-- | Application combinator
(<*>) :: Applicative f => f (a -> b) -> f a -> f b
(<*>) = liftA2 id

-- | Monadic composition
(>>=) :: Monad m => m a -> (a -> m b) -> m b
m >>= f = join (fmap f m)

-- | Kleisli composition
(>=>) :: Monad m => (a -> m b) -> (b -> m c) -> (a -> m c)
f >=> g = \x -> f x >>= g

-- | Fix-point combinator
fix :: (a -> a) -> a
fix f = let x = f x in x

-- | Y combinator implementation
factorial :: Integer -> Integer
factorial = fix $ \f n -> if n <= 1 then 1 else n * f (n - 1)

-- | Memoization using fix-point
memoize :: (Int -> a) -> (Int -> a)
memoize f = (memo M.!)
  where memo = M.fromList $ map (\i -> (i, f i)) [0..1000]

fibMemo :: Int -> Integer
fibMemo = memoize fib
  where
    fib 0 = 0
    fib 1 = 1
    fib n = fibMemo (n-1) + fibMemo (n-2)

-- | Advanced Algorithms
-- ====================

-- | Quick sort with higher-order functions
quickSort :: Ord a => [a] -> [a]
quickSort [] = []
quickSort (x:xs) = 
    let smaller = filter (<= x) xs
        larger = filter (> x) xs
    in quickSort smaller ++ [x] ++ quickSort larger

-- | Merge sort implementation
mergeSort :: Ord a => [a] -> [a]
mergeSort [] = []
mergeSort [x] = [x]
mergeSort xs = 
    let (left, right) = splitAt (length xs `div` 2) xs
    in merge (mergeSort left) (mergeSort right)
  where
    merge [] ys = ys
    merge xs [] = xs
    merge (x:xs) (y:ys)
        | x <= y    = x : merge xs (y:ys)
        | otherwise = y : merge (x:xs) ys

-- | Dijkstra's shortest path algorithm
type Graph = Map Int [(Int, Int)]  -- Vertex -> [(neighbor, weight)]
type Distance = Map Int Int

dijkstra :: Graph -> Int -> Distance
dijkstra graph start = 
    dijkstra' (S.singleton (0, start)) (M.singleton start 0) M.empty
  where
    dijkstra' queue distances visited
        | S.null queue = distances
        | S.member current visited = 
            dijkstra' rest distances visited
        | otherwise = 
            let newDistances = foldl' updateDistance distances neighbors
                newQueue = foldl' addToQueue rest neighbors
                newVisited = M.insert current currentDist visited
            in dijkstra' newQueue newDistances newVisited
      where
        ((currentDist, current), rest) = S.deleteFindMin queue
        neighbors = M.findWithDefault [] current graph
        
        updateDistance dists (neighbor, weight) =
            let newDist = currentDist + weight
                oldDist = M.findWithDefault maxBound neighbor dists
            in if newDist < oldDist
               then M.insert neighbor newDist dists
               else dists
        
        addToQueue q (neighbor, weight) =
            let newDist = currentDist + weight
            in S.insert (newDist, neighbor) q

-- | Category Theory Applications
-- =============================

-- | Functor laws verification
functorLaws :: (Functor f, Eq (f b)) => f a -> (a -> b) -> Bool
functorLaws fa f = 
    fmap id fa == fa &&
    fmap (f . id) fa == fmap f (fmap id fa)

-- | Natural transformation
type Natural f g = forall a. f a -> g a

listToMaybe :: Natural [] Maybe
listToMaybe [] = Nothing
listToMaybe (x:_) = Just x

-- | Monad laws verification
monadLaws :: (Monad m, Eq (m c)) => a -> (a -> m b) -> (b -> m c) -> m a -> Bool
monadLaws a f g ma =
    -- Left identity: return a >>= f == f a
    (return a >>= f) == f a &&
    -- Right identity: m >>= return == m
    (ma >>= return) == ma &&
    -- Associativity: (m >>= f) >>= g == m >>= (\x -> f x >>= g)
    ((ma >>= f) >>= g) == (ma >>= (\x -> f x >>= g))

-- | Free monad implementation
data Free f a = Pure a | Free (f (Free f a))

instance Functor f => Functor (Free f) where
    fmap f (Pure a) = Pure (f a)
    fmap f (Free fa) = Free (fmap (fmap f) fa)

instance Functor f => Applicative (Free f) where
    pure = Pure
    Pure f <*> Pure a = Pure (f a)
    Pure f <*> Free fa = Free (fmap (fmap f) fa)
    Free ff <*> fa = Free (fmap (<*> fa) ff)

instance Functor f => Monad (Free f) where
    return = Pure
    Pure a >>= f = f a
    Free fa >>= f = Free (fmap (>>= f) fa)

-- | Utility Functions and Testing
-- ==============================

-- | Benchmark a function
benchmark :: IO a -> IO (a, NominalDiffTime)
benchmark action = do
    start <- getCurrentTime
    result <- action
    end <- getCurrentTime
    return (result, diffUTCTime end start)

-- | Property-based testing helpers
prop_reverseReverse :: [Int] -> Bool
prop_reverseReverse xs = reverse (reverse xs) == xs

prop_sortIdempotent :: [Int] -> Bool
prop_sortIdempotent xs = 
    let sorted = quickSort xs
    in quickSort sorted == sorted

-- | Test data generators
generateRandomList :: Int -> IO [Int]
generateRandomList n = sequence $ replicate n (randomRIO (1, 100))

-- | Example usage and demonstrations
main :: IO ()
main = do
    putStrLn "=== Advanced Haskell Programming Examples ==="
    
    -- Tree operations
    putStrLn "\n1. Binary Search Tree:"
    let tree = foldl' (flip insertBST) Empty [5, 3, 8, 1, 4, 7, 9]
    putStrLn $ "Tree: " ++ show tree
    putStrLn $ "In-order: " ++ show (inOrderTraversal tree)
    putStrLn $ "Contains 4: " ++ show (searchBST 4 tree)
    
    -- Monadic programming
    putStrLn "\n2. Monadic Application:"
    let config = AppConfig "postgresql://localhost:5432/db" "api-key-123" True
        initialState = AppState 0 0 =<< getCurrentTime
    
    getCurrentTime >>= \time -> do
        (result, finalState) <- runApp config (AppState 0 0 time) $ do
            user1 <- fetchUser 123
            user2 <- fetchUser 456
            return (user1, user2)
        
        putStrLn $ "Final state: " ++ show finalState
    
    -- Expression evaluation
    putStrLn "\n3. GADT Expression Evaluation:"
    let expr = If (Eq (Add (IntLit 2) (IntLit 3)) (IntLit 5))
                  (IntLit 42)
                  (IntLit 0)
    putStrLn $ "Expression result: " ++ show (eval expr)
    
    -- Parallel computation
    putStrLn "\n4. Parallel Computation:"
    let numbers = [1..1000000]
    (result, time) <- benchmark $ return $! parallelSum numbers
    putStrLn $ "Parallel sum: " ++ show result ++ " (computed in " ++ show time ++ ")"
    
    -- Algorithm comparison
    putStrLn "\n5. Sorting Algorithm Comparison:"
    testList <- generateRandomList 10
    putStrLn $ "Original: " ++ show testList
    putStrLn $ "Quick sort: " ++ show (quickSort testList)
    putStrLn $ "Merge sort: " ++ show (mergeSort testList)
    
    -- Property testing
    putStrLn "\n6. Property Testing:"
    testList2 <- generateRandomList 20
    putStrLn $ "Reverse-reverse property: " ++ show (prop_reverseReverse testList2)
    putStrLn $ "Sort idempotent property: " ++ show (prop_sortIdempotent testList2)
    
    putStrLn "\n=== Haskell Demo Complete ==="