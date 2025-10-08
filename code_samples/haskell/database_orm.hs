-- Database: Orm
-- AI/ML Training Sample

module Orm where

data Orm = Orm {
    getData :: String
} deriving (Show, Eq)

process :: Orm -> String -> Orm
process obj input = obj { getData = input }

validate :: Orm -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Orm { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
