-- Functional: Monads
-- AI/ML Training Sample

module Monads where

data Monads = Monads {
    getData :: String
} deriving (Show, Eq)

process :: Monads -> String -> Monads
process obj input = obj { getData = input }

validate :: Monads -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Monads { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
