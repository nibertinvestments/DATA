-- Security: Input Validation
-- AI/ML Training Sample

module InputValidation where

data InputValidation = InputValidation {
    getData :: String
} deriving (Show, Eq)

process :: InputValidation -> String -> InputValidation
process obj input = obj { getData = input }

validate :: InputValidation -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = InputValidation { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
