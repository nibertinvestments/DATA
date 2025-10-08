-- Design Patterns: Decorator
-- AI/ML Training Sample

module Decorator where

data Decorator = Decorator {
    getData :: String
} deriving (Show, Eq)

process :: Decorator -> String -> Decorator
process obj input = obj { getData = input }

validate :: Decorator -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Decorator { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
