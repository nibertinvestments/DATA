-- Data Structures: Linked List
-- AI/ML Training Sample

module LinkedList where

data LinkedList = LinkedList {
    getData :: String
} deriving (Show, Eq)

process :: LinkedList -> String -> LinkedList
process obj input = obj { getData = input }

validate :: LinkedList -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = LinkedList { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
