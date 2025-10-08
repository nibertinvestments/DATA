-- Performance: Memoization
-- AI/ML Training Sample

module Memoization where

data Memoization = Memoization {
    getData :: String
} deriving (Show, Eq)

process :: Memoization -> String -> Memoization
process obj input = obj { getData = input }

validate :: Memoization -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Memoization { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
