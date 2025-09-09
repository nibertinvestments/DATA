// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "./EncryptionA.sol";

contract EncryptionManager {
    address public owner; // Mutable deployer owner
    address public newOwner; // Final owner, immutable via newOwnerSet
    address public backupWallet;
    bytes32 private pinHash;
    uint256 public immutable deploymentTimestamp;
    bool private newOwnerSet;
    mapping(bytes32 => EncryptionA.Ciphertext) private encryptedData;
    mapping(address => bool) public isContractLocked;

    event OwnerSet(address indexed newOwner);
    event PinSet(address indexed newOwner);
    event BackupWalletSet(address indexed backupWallet);
    event ContractLocked(address indexed targetContract, bytes32 indexed dataId);
    event ContractUnlocked(address indexed targetContract, bytes32 indexed dataId);

    constructor() {
        deploymentTimestamp = block.timestamp;
        owner = msg.sender; // Deployer is initial mutable owner
    }

    function setOwner(address _newOwner, uint256 _pin) external {
        require(msg.sender == owner, "Only current owner can set new owner");
        require(!newOwnerSet, "New owner already set");
        require(_newOwner != address(0) && _newOwner.code.length > 0, "New owner must be a contract");
        require(_pin <= 999999, "Invalid PIN: Must be 6 digits");

        newOwnerSet = true;
        newOwner = _newOwner;
        pinHash = keccak256(abi.encodePacked(_pin));
        emit OwnerSet(_newOwner);
        emit PinSet(_newOwner);
    }

    function updateOwner(address _newDeployerOwner) external {
        require(msg.sender == owner, "Only current owner can update owner");
        require(!newOwnerSet, "New owner already set");
        require(_newDeployerOwner != address(0), "Invalid owner address");

        owner = _newDeployerOwner; // Update mutable deployer owner
    }

    function setBackupWallet(address _backupWallet) external {
        require(newOwnerSet, "New owner not set");
        require(msg.sender == newOwner, "Only new owner can set backup wallet");
        require(block.timestamp <= deploymentTimestamp + 12 hours, "Backup window expired");
        require(_backupWallet != address(0), "Invalid backup wallet");
        require(backupWallet == address(0), "Backup wallet already set");

        backupWallet = _backupWallet;
        emit BackupWalletSet(_backupWallet);
    }

    function lockContract(
        address _targetContract,
        bytes32 _dataId,
        bytes32 _data,
        bytes32 _recipientPubKeyX,
        bytes32 _recipientPubKeyY,
        bytes32 _ephemeralPrivKey
    ) external {
        require(msg.sender == owner || (newOwnerSet && msg.sender == newOwner), "Only owner can lock contract");
        require(_targetContract != address(0) && _targetContract.code.length > 0, "Invalid target contract");
        require(encryptedData[_dataId].timestamp == 0, "Data ID already used");
        require(!isContractLocked[_targetContract], "Contract already locked");

        EncryptionA.Ciphertext memory ciphertext = EncryptionA.encrypt(
            _data,
            _recipientPubKeyX,
            _recipientPubKeyY,
            _ephemeralPrivKey
        );
        encryptedData[_dataId] = ciphertext;
        isContractLocked[_targetContract] = true;

        emit ContractLocked(_targetContract, _dataId);
    }

    function unlockContract(
        address _targetContract,
        bytes32 _dataId,
        bytes32 _recipientPrivKey,
        uint256 _pin
    ) external returns (bytes32) {
        require(isContractLocked[_targetContract], "Contract not locked");
        require(encryptedData[_dataId].timestamp != 0, "Data ID not found");
        require(pinHash != bytes32(0), "PIN not set");
        require(newOwnerSet, "New owner not set");

        bytes32 plaintext = EncryptionA.decrypt(
            encryptedData[_dataId],
            _recipientPrivKey,
            _pin,
            pinHash,
            newOwner,
            backupWallet
        );

        isContractLocked[_targetContract] = false;
        delete encryptedData[_dataId];
        emit ContractUnlocked(_targetContract, _dataId);
        return plaintext;
    }
}