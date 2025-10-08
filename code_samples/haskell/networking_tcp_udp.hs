-- Networking: Tcp Udp
-- AI/ML Training Sample

module TcpUdp where

data TcpUdp = TcpUdp {
    getData :: String
} deriving (Show, Eq)

process :: TcpUdp -> String -> TcpUdp
process obj input = obj { getData = input }

validate :: TcpUdp -> Bool
validate obj = not (null (getData obj))

main :: IO ()
main = do
    let instance = TcpUdp { getData = "" }
    let updated = process instance "example"
    putStrLn $ "Data: " ++ getData updated
    putStrLn $ "Valid: " ++ show (validate updated)
