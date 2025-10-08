-- Algorithms: Graph
-- AI/ML Training Sample

module Graph where

data Graph = Graph {
    getData :: String
} deriving (Show, Eq)

process :: Graph -> String -> Graph
process obj input = obj { getData = input }

validate :: Graph -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Graph { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
