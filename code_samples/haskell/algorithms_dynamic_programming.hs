-- Algorithms: Dynamic Programming
-- AI/ML Training Sample

module DynamicProgramming where

data DynamicProgramming = DynamicProgramming {
    getData :: String
} deriving (Show, Eq)

process :: DynamicProgramming -> String -> DynamicProgramming
process obj input = obj { getData = input }

validate :: DynamicProgramming -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = DynamicProgramming { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
