-- Haskell examples for functional programming concepts

{-# LANGUAGE OverloadedStrings #-}

import Data.List (sort, group, nub, intercalate)
import Data.Maybe (fromMaybe, isJust, catMaybes)
import Control.Monad (when, unless, forM_)
import qualified Data.Map as Map
import qualified Data.Set as Set

-- Basic data types and pattern matching
data Shape = Circle Double
           | Rectangle Double Double
           | Triangle Double Double Double
           deriving (Show, Eq)

-- Calculate area using pattern matching
area :: Shape -> Double
area (Circle r) = pi * r * r
area (Rectangle w h) = w * h
area (Triangle a b c) = 
    let s = (a + b + c) / 2
    in sqrt (s * (s - a) * (s - b) * (s - c))

-- Perimeter calculation
perimeter :: Shape -> Double
perimeter (Circle r) = 2 * pi * r
perimeter (Rectangle w h) = 2 * (w + h)
perimeter (Triangle a b c) = a + b + c

-- Algebraic data types for a simple expression language
data Expr = Num Double
          | Var String
          | Add Expr Expr
          | Mul Expr Expr
          | Sub Expr Expr
          | Div Expr Expr
          deriving (Show, Eq)

-- Evaluate expressions with variable substitution
eval :: Map.Map String Double -> Expr -> Maybe Double
eval _ (Num n) = Just n
eval env (Var v) = Map.lookup v env
eval env (Add e1 e2) = (+) <$> eval env e1 <*> eval env e2
eval env (Sub e1 e2) = (-) <$> eval env e1 <*> eval env e2
eval env (Mul e1 e2) = (*) <$> eval env e1 <*> eval env e2
eval env (Div e1 e2) = do
    v1 <- eval env e1
    v2 <- eval env e2
    if v2 == 0 then Nothing else Just (v1 / v2)

-- Tree data structure
data Tree a = Empty
            | Node a (Tree a) (Tree a)
            deriving (Show, Eq)

-- Insert into binary search tree
insertTree :: (Ord a) => a -> Tree a -> Tree a
insertTree x Empty = Node x Empty Empty
insertTree x (Node y left right)
    | x <= y = Node y (insertTree x left) right
    | otherwise = Node y left (insertTree x right)

-- Search in binary search tree
searchTree :: (Ord a) => a -> Tree a -> Bool
searchTree _ Empty = False
searchTree x (Node y left right)
    | x == y = True
    | x < y = searchTree x left
    | otherwise = searchTree x right

-- Tree traversals
inorder :: Tree a -> [a]
inorder Empty = []
inorder (Node x left right) = inorder left ++ [x] ++ inorder right

preorder :: Tree a -> [a]
preorder Empty = []
preorder (Node x left right) = [x] ++ preorder left ++ preorder right

postorder :: Tree a -> [a]
postorder Empty = []
postorder (Node x left right) = postorder left ++ postorder right ++ [x]

-- Higher-order functions and list processing
quicksort :: (Ord a) => [a] -> [a]
quicksort [] = []
quicksort (x:xs) = 
    let smaller = quicksort [y | y <- xs, y <= x]
        larger = quicksort [y | y <- xs, y > x]
    in smaller ++ [x] ++ larger

-- Map implementations
mapList :: (a -> b) -> [a] -> [b]
mapList _ [] = []
mapList f (x:xs) = f x : mapList f xs

-- Filter implementation
filterList :: (a -> Bool) -> [a] -> [a]
filterList _ [] = []
filterList p (x:xs)
    | p x = x : filterList p xs
    | otherwise = filterList p xs

-- Fold implementations
foldlList :: (b -> a -> b) -> b -> [a] -> b
foldlList _ acc [] = acc
foldlList f acc (x:xs) = foldlList f (f acc x) xs

foldrList :: (a -> b -> b) -> b -> [a] -> b
foldrList _ acc [] = acc
foldrList f acc (x:xs) = f x (foldrList f acc xs)

-- Function composition and currying examples
compose :: (b -> c) -> (a -> b) -> a -> c
compose f g x = f (g x)

-- Partial application examples
add :: Int -> Int -> Int
add x y = x + y

addTen :: Int -> Int
addTen = add 10

multiply :: Int -> Int -> Int
multiply x y = x * y

double :: Int -> Int
double = multiply 2

-- Maybe monad usage
safeDivision :: Double -> Double -> Maybe Double
safeDivision _ 0 = Nothing
safeDivision x y = Just (x / y)

safeSquareRoot :: Double -> Maybe Double
safeSquareRoot x
    | x < 0 = Nothing
    | otherwise = Just (sqrt x)

-- Chaining Maybe operations
safeCalculation :: Double -> Double -> Maybe Double
safeCalculation x y = do
    result1 <- safeDivision x y
    result2 <- safeSquareRoot result1
    return (result2 * 2)

-- List comprehensions and generators
pythagoreanTriples :: Int -> [(Int, Int, Int)]
pythagoreanTriples n = [(a, b, c) | a <- [1..n], 
                                    b <- [a..n], 
                                    c <- [b..n], 
                                    a^2 + b^2 == c^2]

-- Prime number generator using Sieve of Eratosthenes
primes :: [Int]
primes = sieve [2..]
  where
    sieve (p:xs) = p : sieve [x | x <- xs, x `mod` p /= 0]

-- Take first n primes
firstNPrimes :: Int -> [Int]
firstNPrimes n = take n primes

-- Fibonacci sequence implementations
-- Naive recursive (inefficient)
fibNaive :: Int -> Integer
fibNaive 0 = 0
fibNaive 1 = 1
fibNaive n = fibNaive (n-1) + fibNaive (n-2)

-- Efficient using infinite list
fibSeq :: [Integer]
fibSeq = 0 : 1 : zipWith (+) fibSeq (tail fibSeq)

fib :: Int -> Integer
fib n = fibSeq !! n

-- String processing functions
wordCount :: String -> Int
wordCount = length . words

charFrequency :: String -> Map.Map Char Int
charFrequency = foldl (\acc c -> Map.insertWith (+) c 1 acc) Map.empty

isPalindrome :: String -> Bool
isPalindrome str = cleanStr == reverse cleanStr
  where cleanStr = map toLower $ filter isAlpha str
        toLower c = if c >= 'A' && c <= 'Z' then toEnum (fromEnum c + 32) else c
        isAlpha c = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')

-- Type classes and polymorphism
class Describable a where
    describe :: a -> String

instance Describable Shape where
    describe (Circle r) = "Circle with radius " ++ show r
    describe (Rectangle w h) = "Rectangle " ++ show w ++ "x" ++ show h
    describe (Triangle a b c) = "Triangle with sides " ++ show a ++ ", " ++ show b ++ ", " ++ show c

instance Describable Int where
    describe n
        | n < 0 = "Negative number: " ++ show n
        | n == 0 = "Zero"
        | otherwise = "Positive number: " ++ show n

-- IO and side effects
processNumbers :: [Int] -> IO ()
processNumbers nums = do
    putStrLn "Processing numbers:"
    forM_ nums $ \n -> do
        putStrLn $ "Number: " ++ show n
        when (even n) $ putStrLn "  -> Even"
        when (odd n) $ putStrLn "  -> Odd"
        when (n > 0) $ putStrLn "  -> Positive"

-- Main function demonstrating all concepts
main :: IO ()
main = do
    putStrLn "Haskell Functional Programming Examples"
    putStrLn "======================================\n"
    
    -- Shape calculations
    let shapes = [Circle 5, Rectangle 4 6, Triangle 3 4 5]
    putStrLn "Shape calculations:"
    mapM_ (\shape -> do
        putStrLn $ describe shape
        putStrLn $ "  Area: " ++ show (area shape)
        putStrLn $ "  Perimeter: " ++ show (perimeter shape)
        ) shapes
    putStrLn ""
    
    -- Expression evaluation
    let expr = Add (Mul (Num 3) (Var "x")) (Num 5)
    let env = Map.fromList [("x", 4), ("y", 2)]
    putStrLn $ "Expression: " ++ show expr
    putStrLn $ "Environment: " ++ show env
    putStrLn $ "Result: " ++ show (eval env expr)
    putStrLn ""
    
    -- Binary search tree
    let tree = foldr insertTree Empty [5, 3, 7, 1, 9, 4, 6]
    putStrLn $ "Binary search tree (inorder): " ++ show (inorder tree)
    putStrLn $ "Search for 4: " ++ show (searchTree 4 tree)
    putStrLn $ "Search for 8: " ++ show (searchTree 8 tree)
    putStrLn ""
    
    -- List operations
    let numbers = [64, 34, 25, 12, 22, 11, 90]
    putStrLn $ "Original list: " ++ show numbers
    putStrLn $ "Sorted: " ++ show (quicksort numbers)
    putStrLn $ "Doubled: " ++ show (mapList (*2) numbers)
    putStrLn $ "Even numbers: " ++ show (filterList even numbers)
    putStrLn $ "Sum: " ++ show (foldlList (+) 0 numbers)
    putStrLn ""
    
    -- Function composition
    let composedFunc = compose (multiply 3) (add 5)
    putStrLn $ "Composed function (3 * (x + 5)) with x=4: " ++ show (composedFunc 4)
    putStrLn $ "Add ten to 15: " ++ show (addTen 15)
    putStrLn $ "Double 7: " ++ show (double 7)
    putStrLn ""
    
    -- Maybe monad
    putStrLn "Safe calculations with Maybe:"
    putStrLn $ "safeCalculation 16 4: " ++ show (safeCalculation 16 4)
    putStrLn $ "safeCalculation 16 0: " ++ show (safeCalculation 16 0)
    putStrLn $ "safeCalculation (-4) 2: " ++ show (safeCalculation (-4) 2)
    putStrLn ""
    
    -- List comprehensions
    putStrLn $ "Pythagorean triples up to 15: " ++ show (pythagoreanTriples 15)
    putStrLn $ "First 10 primes: " ++ show (firstNPrimes 10)
    putStrLn $ "First 10 Fibonacci numbers: " ++ show (take 10 fibSeq)
    putStrLn ""
    
    -- String processing
    let testString = "Racecar"
    putStrLn $ "String: \"" ++ testString ++ "\""
    putStrLn $ "Word count: " ++ show (wordCount testString)
    putStrLn $ "Is palindrome: " ++ show (isPalindrome testString)
    putStrLn $ "Character frequency: " ++ show (charFrequency testString)
    putStrLn ""
    
    -- Process some numbers
    processNumbers [1, 2, 3, 4, 5]
    
    putStrLn "\nAll Haskell examples completed!"