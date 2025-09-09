// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

/**
 * @title MultiSigWallet
 * @dev Multi-signature wallet implementation requiring multiple confirmations for transactions
 */
contract MultiSigWallet {
    struct Transaction {
        address to;
        uint256 value;
        bytes data;
        bool executed;
        uint256 confirmations;
    }
    
    mapping(uint256 => Transaction) public transactions;
    mapping(uint256 => mapping(address => bool)) public confirmations;
    mapping(address => bool) public isOwner;
    
    address[] public owners;
    uint256 public required;
    uint256 public transactionCount;
    
    event Deposit(address indexed sender, uint256 value);
    event SubmitTransaction(address indexed owner, uint256 indexed txIndex, address indexed to, uint256 value, bytes data);
    event ConfirmTransaction(address indexed owner, uint256 indexed txIndex);
    event RevokeConfirmation(address indexed owner, uint256 indexed txIndex);
    event ExecuteTransaction(address indexed owner, uint256 indexed txIndex);
    
    modifier onlyOwner() {
        require(isOwner[msg.sender], "Not an owner");
        _;
    }
    
    modifier txExists(uint256 _txIndex) {
        require(_txIndex < transactionCount, "Transaction does not exist");
        _;
    }
    
    modifier notExecuted(uint256 _txIndex) {
        require(!transactions[_txIndex].executed, "Transaction already executed");
        _;
    }
    
    modifier notConfirmed(uint256 _txIndex) {
        require(!confirmations[_txIndex][msg.sender], "Transaction already confirmed");
        _;
    }
    
    constructor(address[] memory _owners, uint256 _required) {
        require(_owners.length > 0, "Owners required");
        require(_required > 0 && _required <= _owners.length, "Invalid required number of owners");
        
        for (uint256 i = 0; i < _owners.length; i++) {
            address owner = _owners[i];
            require(owner != address(0), "Invalid owner");
            require(!isOwner[owner], "Owner not unique");
            
            isOwner[owner] = true;
            owners.push(owner);
        }
        
        required = _required;
    }
    
    receive() external payable {
        emit Deposit(msg.sender, msg.value);
    }
    
    function submitTransaction(address _to, uint256 _value, bytes memory _data) 
        external 
        onlyOwner 
        returns (uint256 txIndex) 
    {
        txIndex = transactionCount;
        transactions[txIndex] = Transaction({
            to: _to,
            value: _value,
            data: _data,
            executed: false,
            confirmations: 0
        });
        transactionCount++;
        
        emit SubmitTransaction(msg.sender, txIndex, _to, _value, _data);
        
        // Auto-confirm by submitter
        confirmTransaction(txIndex);
    }
    
    function confirmTransaction(uint256 _txIndex) 
        public 
        onlyOwner 
        txExists(_txIndex) 
        notExecuted(_txIndex) 
        notConfirmed(_txIndex) 
    {
        confirmations[_txIndex][msg.sender] = true;
        transactions[_txIndex].confirmations++;
        
        emit ConfirmTransaction(msg.sender, _txIndex);
        
        if (transactions[_txIndex].confirmations >= required) {
            executeTransaction(_txIndex);
        }
    }
    
    function executeTransaction(uint256 _txIndex) 
        public 
        onlyOwner 
        txExists(_txIndex) 
        notExecuted(_txIndex) 
    {
        Transaction storage transaction = transactions[_txIndex];
        require(transaction.confirmations >= required, "Cannot execute transaction");
        
        transaction.executed = true;
        
        (bool success, ) = transaction.to.call{value: transaction.value}(transaction.data);
        require(success, "Transaction failed");
        
        emit ExecuteTransaction(msg.sender, _txIndex);
    }
    
    function revokeConfirmation(uint256 _txIndex) 
        external 
        onlyOwner 
        txExists(_txIndex) 
        notExecuted(_txIndex) 
    {
        require(confirmations[_txIndex][msg.sender], "Transaction not confirmed");
        
        confirmations[_txIndex][msg.sender] = false;
        transactions[_txIndex].confirmations--;
        
        emit RevokeConfirmation(msg.sender, _txIndex);
    }
    
    function getOwners() external view returns (address[] memory) {
        return owners;
    }
    
    function getTransactionCount() external view returns (uint256) {
        return transactionCount;
    }
    
    function getTransaction(uint256 _txIndex) 
        external 
        view 
        returns (
            address to,
            uint256 value,
            bytes memory data,
            bool executed,
            uint256 confirmationCount
        ) 
    {
        Transaction storage transaction = transactions[_txIndex];
        return (
            transaction.to,
            transaction.value,
            transaction.data,
            transaction.executed,
            transaction.confirmations
        );
    }
}