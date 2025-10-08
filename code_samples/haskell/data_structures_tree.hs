-- Data Structures: Tree
-- AI/ML Training Sample

module Tree where

data Tree = Tree {
    getData :: String
} deriving (Show, Eq)

process :: Tree -> String -> Tree
process obj input = obj { getData = input }

validate :: Tree -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Tree { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
