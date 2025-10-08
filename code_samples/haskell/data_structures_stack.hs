-- Data Structures: Stack
-- AI/ML Training Sample

module Stack where

data Stack = Stack {
    getData :: String
} deriving (Show, Eq)

process :: Stack -> String -> Stack
process obj input = obj { getData = input }

validate :: Stack -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Stack { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
