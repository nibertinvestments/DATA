-- Performance: Optimization
-- AI/ML Training Sample

module Optimization where

data Optimization = Optimization {
    getData :: String
} deriving (Show, Eq)

process :: Optimization -> String -> Optimization
process obj input = obj { getData = input }

validate :: Optimization -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Optimization { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
