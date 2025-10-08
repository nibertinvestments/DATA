-- Functional: Higher Order
-- AI/ML Training Sample

module HigherOrder where

data HigherOrder = HigherOrder {
    getData :: String
} deriving (Show, Eq)

process :: HigherOrder -> String -> HigherOrder
process obj input = obj { getData = input }

validate :: HigherOrder -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = HigherOrder { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
