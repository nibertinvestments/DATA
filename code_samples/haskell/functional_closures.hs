-- Functional: Closures
-- AI/ML Training Sample

module Closures where

data Closures = Closures {
    getData :: String
} deriving (Show, Eq)

process :: Closures -> String -> Closures
process obj input = obj { getData = input }

validate :: Closures -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Closures { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
