-- Web Development: Rest Api
-- AI/ML Training Sample

module RestApi where

data RestApi = RestApi {
    getData :: String
} deriving (Show, Eq)

process :: RestApi -> String -> RestApi
process obj input = obj { getData = input }

validate :: RestApi -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = RestApi { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
