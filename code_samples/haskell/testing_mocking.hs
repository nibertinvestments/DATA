-- Testing: Mocking
-- AI/ML Training Sample

module Mocking where

data Mocking = Mocking {
    getData :: String
} deriving (Show, Eq)

process :: Mocking -> String -> Mocking
process obj input = obj { getData = input }

validate :: Mocking -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Mocking { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
