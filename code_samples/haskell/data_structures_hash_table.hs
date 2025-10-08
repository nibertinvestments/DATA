-- Data Structures: Hash Table
-- AI/ML Training Sample

module HashTable where

data HashTable = HashTable {
    getData :: String
} deriving (Show, Eq)

process :: HashTable -> String -> HashTable
process obj input = obj { getData = input }

validate :: HashTable -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = HashTable { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
