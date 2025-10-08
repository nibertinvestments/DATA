-- Design Patterns: Strategy
-- AI/ML Training Sample

module Strategy where

data Strategy = Strategy {
    getData :: String
} deriving (Show, Eq)

process :: Strategy -> String -> Strategy
process obj input = obj { getData = input }

validate :: Strategy -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Strategy { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
