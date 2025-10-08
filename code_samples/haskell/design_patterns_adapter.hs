-- Design Patterns: Adapter
-- AI/ML Training Sample

module Adapter where

data Adapter = Adapter {
    getData :: String
} deriving (Show, Eq)

process :: Adapter -> String -> Adapter
process obj input = obj { getData = input }

validate :: Adapter -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Adapter { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
