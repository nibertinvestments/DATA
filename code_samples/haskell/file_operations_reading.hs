-- File Operations: Reading
-- AI/ML Training Sample

module Reading where

data Reading = Reading {
    getData :: String
} deriving (Show, Eq)

process :: Reading -> String -> Reading
process obj input = obj { getData = input }

validate :: Reading -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Reading { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
