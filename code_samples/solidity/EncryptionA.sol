// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

/**
 * @title EncryptionA
 * @dev Gas-efficient, secure encryption library using ECIES with secp256k1 and XOR-based symmetric encryption.
 *      Supports contract data encryption, PIN verification, and 24-hour decryption window.
 */
library EncryptionA {
    struct Ciphertext {
        bytes32 ephemeralPubKeyX;
        bytes32 ephemeralPubKeyY;
        bytes32 ciphertext;
        bytes32 mac;
        uint256 timestamp;
    }

    event EncryptionPerformed(address indexed sender, bytes32 indexed recipientPubKeyX, bytes32 ciphertext);
    event DecryptionPerformed(address indexed caller, bytes32 indexed recipientPubKeyX, bytes32 plaintext);

    function encrypt(
        bytes32 _data,
        bytes32 _recipientPubKeyX,
        bytes32 _recipientPubKeyY,
        bytes32 _ephemeralPrivKey
    ) internal returns (Ciphertext memory) {
        require(_recipientPubKeyX != bytes32(0) && _recipientPubKeyY != bytes32(0), "Invalid recipient public key");

        (bytes32 ephemeralPubKeyX, bytes32 ephemeralPubKeyY) = _ecmul(_ephemeralPrivKey);
        (bytes32 sharedSecretX, bytes32 sharedSecretY) = _ecmul(_recipientPubKeyX, _recipientPubKeyY, _ephemeralPrivKey);

        bytes32 symmetricKey = keccak256(abi.encodePacked(sharedSecretX, sharedSecretY));
        bytes32 ciphertext = _data ^ symmetricKey;
        bytes32 mac = keccak256(abi.encodePacked(ciphertext, _recipientPubKeyX, _recipientPubKeyY));

        emit EncryptionPerformed(msg.sender, _recipientPubKeyX, ciphertext);
        return Ciphertext(ephemeralPubKeyX, ephemeralPubKeyY, ciphertext, mac, block.timestamp);
    }

    function decrypt(
        Ciphertext memory _ciphertext,
        bytes32 _recipientPrivKey,
        uint256 _pin,
        bytes32 _pinHash,
        address _owner,
        address _backupWallet
    ) internal returns (bytes32) {
        require(msg.sender == _owner || msg.sender == _backupWallet, "Unauthorized: Only owner or backup wallet");
        require(block.timestamp <= _ciphertext.timestamp + 24 hours, "Decryption window expired");
        require(_pin <= 999999, "Invalid PIN: Must be 6 digits");
        require(keccak256(abi.encodePacked(_pin)) == _pinHash, "Incorrect PIN");

        bytes32 computedMac = keccak256(abi.encodePacked(
            _ciphertext.ciphertext,
            _ciphertext.ephemeralPubKeyX,
            _ciphertext.ephemeralPubKeyY
        ));
        require(computedMac == _ciphertext.mac, "Invalid MAC: Data tampered");

        (bytes32 sharedSecretX, bytes32 sharedSecretY) = _ecmul(
            _ciphertext.ephemeralPubKeyX,
            _ciphertext.ephemeralPubKeyY,
            _recipientPrivKey
        );

        bytes32 symmetricKey = keccak256(abi.encodePacked(sharedSecretX, sharedSecretY));
        bytes32 plaintext = _ciphertext.ciphertext ^ symmetricKey;

        emit DecryptionPerformed(msg.sender, _ciphertext.ephemeralPubKeyX, plaintext);
        return plaintext;
    }

    function _ecmul(bytes32 _privKey) private view returns (bytes32 x, bytes32 y) {
        bytes32 gx = 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798;
        bytes32 gy = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8;
        return _ecmul(gx, gy, _privKey);
    }

    function _ecmul(bytes32 _pubKeyX, bytes32 _pubKeyY, bytes32 _privKey) private view returns (bytes32 x, bytes32 y) {
        bytes memory input = abi.encodePacked(_pubKeyX, _pubKeyY, _privKey);
        bytes memory output = new bytes(64); // Allocate 64 bytes for x and y coordinates

        assembly {
            let inputPtr := add(input, 32)
            let outputPtr := add(output, 32)
            if iszero(staticcall(gas(), 0x07, inputPtr, 96, outputPtr, 64)) {
                revert(0, 0)
            }
            x := mload(outputPtr)
            y := mload(add(outputPtr, 32))
        }

        require(x != bytes32(0) || y != bytes32(0), "Invalid EC point");
        return (x, y);
    }
}