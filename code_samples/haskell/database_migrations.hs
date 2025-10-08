-- Database: Migrations
-- AI/ML Training Sample

module Migrations where

data Migrations = Migrations {
    getData :: String
} deriving (Show, Eq)

process :: Migrations -> String -> Migrations
process obj input = obj { getData = input }

validate :: Migrations -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Migrations { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
