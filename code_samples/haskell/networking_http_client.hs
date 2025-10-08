-- Networking: Http Client
-- AI/ML Training Sample

module HttpClient where

data HttpClient = HttpClient {
    getData :: String
} deriving (Show, Eq)

process :: HttpClient -> String -> HttpClient
process obj input = obj { getData = input }

validate :: HttpClient -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = HttpClient { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
