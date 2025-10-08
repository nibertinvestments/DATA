-- Utilities: Regex
-- AI/ML Training Sample

module Regex where

data Regex = Regex {
    getData :: String
} deriving (Show, Eq)

process :: Regex -> String -> Regex
process obj input = obj { getData = input }

validate :: Regex -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Regex { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
