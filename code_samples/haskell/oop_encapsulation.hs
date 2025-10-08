-- Oop: Encapsulation
-- AI/ML Training Sample

module Encapsulation where

data Encapsulation = Encapsulation {
    getData :: String
} deriving (Show, Eq)

process :: Encapsulation -> String -> Encapsulation
process obj input = obj { getData = input }

validate :: Encapsulation -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Encapsulation { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
