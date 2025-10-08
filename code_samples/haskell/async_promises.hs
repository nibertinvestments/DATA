-- Async: Promises
-- AI/ML Training Sample

module Promises where

data Promises = Promises {
    getData :: String
} deriving (Show, Eq)

process :: Promises -> String -> Promises
process obj input = obj { getData = input }

validate :: Promises -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Promises { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
