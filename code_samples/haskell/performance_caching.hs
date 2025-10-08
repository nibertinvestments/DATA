-- Performance: Caching
-- AI/ML Training Sample

module Caching where

data Caching = Caching {
    getData :: String
} deriving (Show, Eq)

process :: Caching -> String -> Caching
process obj input = obj { getData = input }

validate :: Caching -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Caching { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
