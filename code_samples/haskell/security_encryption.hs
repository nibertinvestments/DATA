-- Security: Encryption
-- AI/ML Training Sample

module Encryption where

data Encryption = Encryption {
    getData :: String
} deriving (Show, Eq)

process :: Encryption -> String -> Encryption
process obj input = obj { getData = input }

validate :: Encryption -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Encryption { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
