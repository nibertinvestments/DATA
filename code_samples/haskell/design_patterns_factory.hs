-- Design Patterns: Factory
-- AI/ML Training Sample

module Factory where

data Factory = Factory {
    getData :: String
} deriving (Show, Eq)

process :: Factory -> String -> Factory
process obj input = obj { getData = input }

validate :: Factory -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Factory { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
