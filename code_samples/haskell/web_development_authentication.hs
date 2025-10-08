-- Web Development: Authentication
-- AI/ML Training Sample

module Authentication where

data Authentication = Authentication {
    getData :: String
} deriving (Show, Eq)

process :: Authentication -> String -> Authentication
process obj input = obj { getData = input }

validate :: Authentication -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Authentication { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
