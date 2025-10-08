-- Utilities: String Manipulation
-- AI/ML Training Sample

module StringManipulation where

data StringManipulation = StringManipulation {
    getData :: String
} deriving (Show, Eq)

process :: StringManipulation -> String -> StringManipulation
process obj input = obj { getData = input }

validate :: StringManipulation -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = StringManipulation { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
