-- Async: Coroutines
-- AI/ML Training Sample

module Coroutines where

data Coroutines = Coroutines {
    getData :: String
} deriving (Show, Eq)

process :: Coroutines -> String -> Coroutines
process obj input = obj { getData = input }

validate :: Coroutines -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Coroutines { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
