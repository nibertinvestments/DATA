-- Web Development: Validation
-- AI/ML Training Sample

module Validation where

data Validation = Validation {
    getData :: String
} deriving (Show, Eq)

process :: Validation -> String -> Validation
process obj input = obj { getData = input }

validate :: Validation -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Validation { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
