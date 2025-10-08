-- File Operations: Writing
-- AI/ML Training Sample

module Writing where

data Writing = Writing {
    getData :: String
} deriving (Show, Eq)

process :: Writing -> String -> Writing
process obj input = obj { getData = input }

validate :: Writing -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Writing { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
