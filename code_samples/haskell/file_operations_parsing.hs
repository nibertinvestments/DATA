-- File Operations: Parsing
-- AI/ML Training Sample

module Parsing where

data Parsing = Parsing {
    getData :: String
} deriving (Show, Eq)

process :: Parsing -> String -> Parsing
process obj input = obj { getData = input }

validate :: Parsing -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Parsing { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
