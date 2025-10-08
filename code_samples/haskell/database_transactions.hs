-- Database: Transactions
-- AI/ML Training Sample

module Transactions where

data Transactions = Transactions {
    getData :: String
} deriving (Show, Eq)

process :: Transactions -> String -> Transactions
process obj input = obj { getData = input }

validate :: Transactions -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Transactions { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
