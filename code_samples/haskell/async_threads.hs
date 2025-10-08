-- Async: Threads
-- AI/ML Training Sample

module Threads where

data Threads = Threads {
    getData :: String
} deriving (Show, Eq)

process :: Threads -> String -> Threads
process obj input = obj { getData = input }

validate :: Threads -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Threads { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
