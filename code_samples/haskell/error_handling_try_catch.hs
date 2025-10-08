-- Error Handling: Try Catch
-- AI/ML Training Sample

module TryCatch where

data TryCatch = TryCatch {
    getData :: String
} deriving (Show, Eq)

process :: TryCatch -> String -> TryCatch
process obj input = obj { getData = input }

validate :: TryCatch -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = TryCatch { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
