-- Oop: Abstraction
-- AI/ML Training Sample

module Abstraction where

data Abstraction = Abstraction {
    getData :: String
} deriving (Show, Eq)

process :: Abstraction -> String -> Abstraction
process obj input = obj { getData = input }

validate :: Abstraction -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Abstraction { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
