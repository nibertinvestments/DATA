-- Async: Async Await
-- AI/ML Training Sample

module AsyncAwait where

data AsyncAwait = AsyncAwait {
    getData :: String
} deriving (Show, Eq)

process :: AsyncAwait -> String -> AsyncAwait
process obj input = obj { getData = input }

validate :: AsyncAwait -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = AsyncAwait { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
