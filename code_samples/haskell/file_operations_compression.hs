-- File Operations: Compression
-- AI/ML Training Sample

module Compression where

data Compression = Compression {
    getData :: String
} deriving (Show, Eq)

process :: Compression -> String -> Compression
process obj input = obj { getData = input }

validate :: Compression -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Compression { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
