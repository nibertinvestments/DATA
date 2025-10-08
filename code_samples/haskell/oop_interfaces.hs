-- Oop: Interfaces
-- AI/ML Training Sample

module Interfaces where

data Interfaces = Interfaces {
    getData :: String
} deriving (Show, Eq)

process :: Interfaces -> String -> Interfaces
process obj input = obj { getData = input }

validate :: Interfaces -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Interfaces { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
