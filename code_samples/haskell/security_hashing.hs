-- Security: Hashing
-- AI/ML Training Sample

module Hashing where

data Hashing = Hashing {
    getData :: String
} deriving (Show, Eq)

process :: Hashing -> String -> Hashing
process obj input = obj { getData = input }

validate :: Hashing -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Hashing { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
