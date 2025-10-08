-- Design Patterns: Observer
-- AI/ML Training Sample

module Observer where

data Observer = Observer {
    getData :: String
} deriving (Show, Eq)

process :: Observer -> String -> Observer
process obj input = obj { getData = input }

validate :: Observer -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Observer { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
