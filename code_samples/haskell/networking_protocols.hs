-- Networking: Protocols
-- AI/ML Training Sample

module Protocols where

data Protocols = Protocols {
    getData :: String
} deriving (Show, Eq)

process :: Protocols -> String -> Protocols
process obj input = obj { getData = input }

validate :: Protocols -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Protocols { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
