-- Utilities: Date Time
-- AI/ML Training Sample

module DateTime where

data DateTime = DateTime {
    getData :: String
} deriving (Show, Eq)

process :: DateTime -> String -> DateTime
process obj input = obj { getData = input }

validate :: DateTime -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = DateTime { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
