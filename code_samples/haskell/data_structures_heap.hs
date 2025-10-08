-- Data Structures: Heap
-- AI/ML Training Sample

module Heap where

data Heap = Heap {
    getData :: String
} deriving (Show, Eq)

process :: Heap -> String -> Heap
process obj input = obj { getData = input }

validate :: Heap -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Heap { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
