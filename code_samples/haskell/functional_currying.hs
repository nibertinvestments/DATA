-- Functional: Currying
-- AI/ML Training Sample

module Currying where

data Currying = Currying {
    getData :: String
} deriving (Show, Eq)

process :: Currying -> String -> Currying
process obj input = obj { getData = input }

validate :: Currying -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Currying { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
