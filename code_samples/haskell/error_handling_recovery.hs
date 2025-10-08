-- Error Handling: Recovery
-- AI/ML Training Sample

module Recovery where

data Recovery = Recovery {
    getData :: String
} deriving (Show, Eq)

process :: Recovery -> String -> Recovery
process obj input = obj { getData = input }

validate :: Recovery -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Recovery { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
