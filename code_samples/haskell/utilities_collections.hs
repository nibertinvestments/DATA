-- Utilities: Collections
-- AI/ML Training Sample

module Collections where

data Collections = Collections {
    getData :: String
} deriving (Show, Eq)

process :: Collections -> String -> Collections
process obj input = obj { getData = input }

validate :: Collections -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Collections { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
