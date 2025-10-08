-- Web Development: Routing
-- AI/ML Training Sample

module Routing where

data Routing = Routing {
    getData :: String
} deriving (Show, Eq)

process :: Routing -> String -> Routing
process obj input = obj { getData = input }

validate :: Routing -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Routing { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
