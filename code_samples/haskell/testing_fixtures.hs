-- Testing: Fixtures
-- AI/ML Training Sample

module Fixtures where

data Fixtures = Fixtures {
    getData :: String
} deriving (Show, Eq)

process :: Fixtures -> String -> Fixtures
process obj input = obj { getData = input }

validate :: Fixtures -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Fixtures { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
