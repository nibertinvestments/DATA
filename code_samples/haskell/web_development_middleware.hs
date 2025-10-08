-- Web Development: Middleware
-- AI/ML Training Sample

module Middleware where

data Middleware = Middleware {
    getData :: String
} deriving (Show, Eq)

process :: Middleware -> String -> Middleware
process obj input = obj { getData = input }

validate :: Middleware -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Middleware { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
