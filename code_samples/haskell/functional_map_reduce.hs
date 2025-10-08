-- Functional: Map Reduce
-- AI/ML Training Sample

module MapReduce where

data MapReduce = MapReduce {
    getData :: String
} deriving (Show, Eq)

process :: MapReduce -> String -> MapReduce
process obj input = obj { getData = input }

validate :: MapReduce -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = MapReduce { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
