-- Testing: Unit Tests
-- AI/ML Training Sample

module UnitTests where

data UnitTests = UnitTests {
    getData :: String
} deriving (Show, Eq)

process :: UnitTests -> String -> UnitTests
process obj input = obj { getData = input }

validate :: UnitTests -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = UnitTests { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
