-- Data Structures: Trie
-- AI/ML Training Sample

module Trie where

data Trie = Trie {
    getData :: String
} deriving (Show, Eq)

process :: Trie -> String -> Trie
process obj input = obj { getData = input }

validate :: Trie -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Trie { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
