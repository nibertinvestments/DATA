// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "./EncryptionA.sol";

interface IEncryptionManager {
    function lockContract(address _targetContract, bytes32 _dataId, bytes32 _data, bytes32 _recipientPubKeyX, bytes32 _recipientPubKeyY, bytes32 _ephemeralPrivKey) external;
    function unlockContract(address _targetContract, bytes32 _dataId, bytes32 _recipientPrivKey, uint256 _pin) external returns (bytes32);
    function isContractLocked(address _targetContract) external view returns (bool);
}

interface IWorkingLockManager {
    function lockInterface(address _targetContract, bytes32 _dataId, bytes32 _data, bytes32 _recipientPubKeyX, bytes32 _recipientPubKeyY, bytes32 _ephemeralPrivKey) external;
    function unlockInterface(address _targetContract, bytes32 _dataId, bytes32 _recipientPrivKey, uint256 _pin) external returns (bytes32);
    function triggerFullLockdown(address _targetContract, bytes32 _dataId, bytes32 _data, bytes32 _recipientPubKeyX, bytes32 _recipientPubKeyY, bytes32 _ephemeralPrivKey) external;
    function isInterfaceLocked(address _targetContract) external view returns (bool);
}

contract LockableContract {
    IWorkingLockManager public workingLockManager;
    IEncryptionManager public encryptionManager;

    modifier notLocked() {
        require(
            !workingLockManager.isInterfaceLocked(address(this)) &&
            !encryptionManager.isContractLocked(address(this)),
            "Contract interface is locked"
        );
        _;
    }

    modifier allowInternal() {
        require(
            !encryptionManager.isContractLocked(address(this)),
            "Contract is fully locked"
        );
        _;
    }

    constructor(address _workingLockManager, address _encryptionManager) {
        workingLockManager = IWorkingLockManager(_workingLockManager);
        encryptionManager = IEncryptionManager(_encryptionManager);
    }

    function triggerInterfaceLock(
        bytes32 _dataId,
        bytes32 _data,
        bytes32 _recipientPubKeyX,
        bytes32 _recipientPubKeyY,
        bytes32 _ephemeralPrivKey
    ) external {
        workingLockManager.lockInterface(address(this), _dataId, _data, _recipientPubKeyX, _recipientPubKeyY, _ephemeralPrivKey);
    }

    function triggerFullLockdown(
        bytes32 _dataId,
        bytes32 _data,
        bytes32 _recipientPubKeyX,
        bytes32 _recipientPubKeyY,
        bytes32 _ephemeralPrivKey
    ) external {
        workingLockManager.triggerFullLockdown(address(this), _dataId, _data, _recipientPubKeyX, _recipientPubKeyY, _ephemeralPrivKey);
    }

    function unlockInterface(
        bytes32 _dataId,
        bytes32 _recipientPrivKey,
        uint256 _pin
    ) external returns (bytes32) {
        return workingLockManager.unlockInterface(address(this), _dataId, _recipientPrivKey, _pin);
    }

    function unlockContract(
        bytes32 _dataId,
        bytes32 _recipientPrivKey,
        uint256 _pin
    ) external returns (bytes32) {
        return encryptionManager.unlockContract(address(this), _dataId, _recipientPrivKey, _pin);
    }

    function externalFunction() external notLocked {
        // External logic
    }

    function internalFunction() internal allowInternal {
        // Internal logic
    }

    function viewFunction() external view allowInternal returns (uint256) {
        return 42;
    }
}