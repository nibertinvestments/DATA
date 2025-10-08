-- Algorithms: Sorting
-- AI/ML Training Sample

module Sorting where

data Sorting = Sorting {
    getData :: String
} deriving (Show, Eq)

process :: Sorting -> String -> Sorting
process obj input = obj { getData = input }

validate :: Sorting -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Sorting { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
