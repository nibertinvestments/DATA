-- Error Handling: Error Propagation
-- AI/ML Training Sample

module ErrorPropagation where

data ErrorPropagation = ErrorPropagation {
    getData :: String
} deriving (Show, Eq)

process :: ErrorPropagation -> String -> ErrorPropagation
process obj input = obj { getData = input }

validate :: ErrorPropagation -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = ErrorPropagation { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
