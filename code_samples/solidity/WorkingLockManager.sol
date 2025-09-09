// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

import "./EncryptionA.sol";
import "./EncryptionManager.sol";

/**
 * @title WorkingLockManager
 * @dev Manages interface locking with owner-controlled security protocol.
 *      Allows partial interface lock or full lockdown via EncryptionManager.
 *      Note: Uses block.timestamp for timelocks, which may vary due to block time fluctuations.
 */
contract WorkingLockManager {
    address public owner; // Mutable deployer owner
    address public newOwner; // Final owner, immutable via newOwnerNotSet
    address public _backupWallet;
    bytes32 private _pinHash;
    uint48 public immutable deploymentTimestamp; // Use uint48
    bool private newOwnerNotSet; // Inverse logic for gas efficiency
    mapping(bytes32 dataId => EncryptionA.Ciphertext ciphertext) private _encryptedData;
    mapping(address targetContract => bool isLocked) public _isInterfaceLocked;
    EncryptionManager public immutable encryptionManager;

    event OwnerUpdated(address indexed newOwner);
    event OwnerSet(address indexed newOwner);
    event PinSet(address indexed newOwner);
    event BackupWalletSet(address indexed backupWallet);
    event InterfaceLocked(address indexed targetContract, bytes32 indexed dataId);
    event InterfaceUnlocked(address indexed targetContract, bytes32 indexed dataId);
    event FullLockdownTriggered(address indexed targetContract, bytes32 indexed dataId);

    modifier onlyOwner() {
        require(msg.sender == owner || (newOwner != address(0) && msg.sender == newOwner), "Only owner");
        _;
    }

    constructor(address _encryptionManager) payable {
        require(_encryptionManager != address(0), "Invalid manager");
        require(_encryptionManager.code.length > 0, "Not a contract");
        deploymentTimestamp = uint48(block.timestamp);
        owner = msg.sender;
        encryptionManager = EncryptionManager(_encryptionManager);
        newOwnerNotSet = true; // Avoid zero-to-one write
    }

    function setOwner(address _newOwner, uint256 _pin) external onlyOwner {
        require(newOwnerNotSet, "New owner set");
        require(_newOwner != address(0), "Invalid new owner");
        require(_newOwner.code.length > 0, "Not a contract");
        require(_pin <= 999999, "Invalid PIN: 6 digits max");

        if (newOwner != _newOwner) {
            newOwner = _newOwner;
            newOwnerNotSet = false;
        }
        bytes32 newPinHash = keccak256(abi.encode(_pin));
        if (_pinHash != newPinHash) {
            _pinHash = newPinHash;
        }
        emit OwnerSet(_newOwner);
        emit PinSet(_newOwner);
    }

    function updateOwner(address _newDeployerOwner) external onlyOwner {
        require(newOwnerNotSet, "New owner set");
        require(_newDeployerOwner != address(0), "Invalid owner");
        if (owner != _newDeployerOwner) {
            owner = _newDeployerOwner;
            emit OwnerUpdated(_newDeployerOwner);
        }
    }

    function setBackupWallet(address _newBackupWallet) external onlyOwner {
        require(!newOwnerNotSet, "New owner not set");
        require(block.timestamp < deploymentTimestamp + 12 hours, "Backup window expired");
        require(_newBackupWallet != address(0), "Invalid backup wallet");
        require(_backupWallet == address(0), "Backup wallet set");

        _backupWallet = _newBackupWallet;
        emit BackupWalletSet(_newBackupWallet);
    }

    function lockInterface(
        address _targetContract,
        bytes32 _dataId,
        bytes32 _data,
        bytes32 _recipientPubKeyX,
        bytes32 _recipientPubKeyY,
        bytes32 _ephemeralPrivKey
    ) external onlyOwner {
        require(_targetContract != address(0), "Invalid target");
        require(_targetContract.code.length > 0, "Not a contract");
        require(_encryptedData[_dataId].timestamp == 0, "Data ID used");
        require(!_isInterfaceLocked[_targetContract], "Interface locked");

        EncryptionA.Ciphertext memory ciphertext = EncryptionA.encrypt(
            _data,
            _recipientPubKeyX,
            _recipientPubKeyY,
            _ephemeralPrivKey
        );
        _encryptedData[_dataId] = ciphertext;
        _isInterfaceLocked[_targetContract] = true;

        emit InterfaceLocked(_targetContract, _dataId);
    }

    function unlockInterface(
        address _targetContract,
        bytes32 _dataId,
        bytes32 _recipientPrivKey,
        uint256 _pin
    ) external onlyOwner returns (bytes32 plaintext) {
        require(_isInterfaceLocked[_targetContract], "Not locked");
        require(_encryptedData[_dataId].timestamp != 0, "Data ID not found");
        require(_pinHash != bytes32(0), "PIN not set");
        require(!newOwnerNotSet, "New owner not set");

        EncryptionA.Ciphertext memory ciphertext = _encryptedData[_dataId]; // Cache storage
        plaintext = EncryptionA.decrypt(
            ciphertext,
            _recipientPrivKey,
            _pin,
            _pinHash,
            newOwner,
            _backupWallet
        );

        _isInterfaceLocked[_targetContract] = false;
        delete _encryptedData[_dataId]; // Use delete for gas refund
        emit InterfaceUnlocked(_targetContract, _dataId);
    }

    function triggerFullLockdown(
        address _targetContract,
        bytes32 _dataId,
        bytes32 _data,
        bytes32 _recipientPubKeyX,
        bytes32 _recipientPubKeyY,
        bytes32 _ephemeralPrivKey
    ) external onlyOwner {
        require(_targetContract != address(0), "Invalid target");
        require(_targetContract.code.length > 0, "Not a contract");
        require(!encryptionManager.isContractLocked(_targetContract), "Contract locked");

        encryptionManager.lockContract(_targetContract, _dataId, _data, _recipientPubKeyX, _recipientPubKeyY, _ephemeralPrivKey);
        _isInterfaceLocked[_targetContract] = true;
        emit FullLockdownTriggered(_targetContract, _dataId);
    }
}