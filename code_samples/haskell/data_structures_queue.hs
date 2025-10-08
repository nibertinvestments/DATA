-- Data Structures: Queue
-- AI/ML Training Sample

module Queue where

data Queue = Queue {
    getData :: String
} deriving (Show, Eq)

process :: Queue -> String -> Queue
process obj input = obj { getData = input }

validate :: Queue -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = Queue { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
