-- Performance: Lazy Loading
-- AI/ML Training Sample

module LazyLoading where

data LazyLoading = LazyLoading {
    getData :: String
} deriving (Show, Eq)

process :: LazyLoading -> String -> LazyLoading
process obj input = obj { getData = input }

validate :: LazyLoading -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = LazyLoading { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
