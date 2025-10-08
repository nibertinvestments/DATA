-- Algorithms: Searching
-- AI/ML Training Sample

module Searching where

data Searching = Searching {
    getData :: String
} deriving (Show, Eq)

process :: Searching -> String -> Searching
process obj input = obj { getData = input }

validate :: Searching -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Searching { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
