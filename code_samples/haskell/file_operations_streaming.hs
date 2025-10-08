-- File Operations: Streaming
-- AI/ML Training Sample

module Streaming where

data Streaming = Streaming {
    getData :: String
} deriving (Show, Eq)

process :: Streaming -> String -> Streaming
process obj input = obj { getData = input }

validate :: Streaming -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Streaming { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
