-- Networking: Websockets
-- AI/ML Training Sample

module Websockets where

data Websockets = Websockets {
    getData :: String
} deriving (Show, Eq)

process :: Websockets -> String -> Websockets
process obj input = obj { getData = input }

validate :: Websockets -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Websockets { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
