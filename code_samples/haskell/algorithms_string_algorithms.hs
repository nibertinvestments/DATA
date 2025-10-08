-- Algorithms: String Algorithms
-- AI/ML Training Sample

module StringAlgorithms where

data StringAlgorithms = StringAlgorithms {
    getData :: String
} deriving (Show, Eq)

process :: StringAlgorithms -> String -> StringAlgorithms
process obj input = obj { getData = input }

validate :: StringAlgorithms -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = StringAlgorithms { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
