-- Networking: Socket Programming
-- AI/ML Training Sample

module SocketProgramming where

data SocketProgramming = SocketProgramming {
    getData :: String
} deriving (Show, Eq)

process :: SocketProgramming -> String -> SocketProgramming
process obj input = obj { getData = input }

validate :: SocketProgramming -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = SocketProgramming { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
