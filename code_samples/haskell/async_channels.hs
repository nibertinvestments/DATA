-- Async: Channels
-- AI/ML Training Sample

module Channels where

data Channels = Channels {
    getData :: String
} deriving (Show, Eq)

process :: Channels -> String -> Channels
process obj input = obj { getData = input }

validate :: Channels -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Channels { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
