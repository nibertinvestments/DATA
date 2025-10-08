-- Oop: Inheritance
-- AI/ML Training Sample

module Inheritance where

data Inheritance = Inheritance {
    getData :: String
} deriving (Show, Eq)

process :: Inheritance -> String -> Inheritance
process obj input = obj { getData = input }

validate :: Inheritance -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Inheritance { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
