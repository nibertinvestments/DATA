-- Security: Authorization
-- AI/ML Training Sample

module Authorization where

data Authorization = Authorization {
    getData :: String
} deriving (Show, Eq)

process :: Authorization -> String -> Authorization
process obj input = obj { getData = input }

validate :: Authorization -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Authorization { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
