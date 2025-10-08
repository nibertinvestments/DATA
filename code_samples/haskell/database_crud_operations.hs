-- Database: Crud Operations
-- AI/ML Training Sample

module CrudOperations where

data CrudOperations = CrudOperations {
    getData :: String
} deriving (Show, Eq)

process :: CrudOperations -> String -> CrudOperations
process obj input = obj { getData = input }

validate :: CrudOperations -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = CrudOperations { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
