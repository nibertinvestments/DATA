-- Performance: Batching
-- AI/ML Training Sample

module Batching where

data Batching = Batching {
    getData :: String
} deriving (Show, Eq)

process :: Batching -> String -> Batching
process obj input = obj { getData = input }

validate :: Batching -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Batching { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
