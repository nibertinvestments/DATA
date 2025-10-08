-- Error Handling: Logging
-- AI/ML Training Sample

module Logging where

data Logging = Logging {
    getData :: String
} deriving (Show, Eq)

process :: Logging -> String -> Logging
process obj input = obj { getData = input }

validate :: Logging -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Logging { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
