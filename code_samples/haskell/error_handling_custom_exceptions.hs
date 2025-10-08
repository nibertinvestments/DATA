-- Error Handling: Custom Exceptions
-- AI/ML Training Sample

module CustomExceptions where

data CustomExceptions = CustomExceptions {
    getData :: String
} deriving (Show, Eq)

process :: CustomExceptions -> String -> CustomExceptions
process obj input = obj { getData = input }

validate :: CustomExceptions -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = CustomExceptions { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
