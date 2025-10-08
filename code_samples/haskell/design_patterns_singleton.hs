-- Design Patterns: Singleton
-- AI/ML Training Sample

module Singleton where

data Singleton = Singleton {
    getData :: String
} deriving (Show, Eq)

process :: Singleton -> String -> Singleton
process obj input = obj { getData = input }

validate :: Singleton -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Singleton { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
