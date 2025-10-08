-- Oop: Polymorphism
-- AI/ML Training Sample

module Polymorphism where

data Polymorphism = Polymorphism {
    getData :: String
} deriving (Show, Eq)

process :: Polymorphism -> String -> Polymorphism
process obj input = obj { getData = input }

validate :: Polymorphism -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Polymorphism { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
