-- Database: Query Builder
-- AI/ML Training Sample

module QueryBuilder where

data QueryBuilder = QueryBuilder {
    getData :: String
} deriving (Show, Eq)

process :: QueryBuilder -> String -> QueryBuilder
process obj input = obj { getData = input }

validate :: QueryBuilder -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = QueryBuilder { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
