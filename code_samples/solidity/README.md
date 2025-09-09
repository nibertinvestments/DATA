# Solidity Code Examples

This directory contains comprehensive Solidity smart contract examples for ML/AI training datasets. The contracts demonstrate various patterns, security practices, and blockchain development concepts.

## Contract Categories

### Core Contracts (from nibertinvestments/contracts)
- **EncryptionA.sol** - Gas-efficient encryption library using ECIES with secp256k1
- **EncryptionManager.sol** - Contract encryption management system with PIN verification
- **LockableContract.sol** - Contract with sophisticated locking mechanisms
- **ReflectionLibrary.sol** - Comprehensive ERC-20 reflection rewards library (19.8KB)
- **WorkingLockManager.sol** - Interface locking management with owner-controlled security

### Basic Contract Examples
- **SimpleToken.sol** - Basic ERC-20 token with minting and burning
- **SimpleVoting.sol** - Basic voting contract with proposal management
- **MultiSigWallet.sol** - Multi-signature wallet requiring multiple confirmations
- **SimpleNFT.sol** - Basic ERC-721 NFT implementation with metadata

## Key Features Demonstrated

### Security Patterns
- Access control with owner/admin roles
- Multi-signature requirements
- Time-based restrictions (24-hour windows)
- PIN verification systems
- Input validation and require statements

### Gas Optimization
- Efficient storage layouts
- Unchecked arithmetic where safe
- Delete operations for gas refunds
- Batch operations to reduce transaction costs

### Advanced Solidity Features
- Libraries vs Contracts
- Modifiers for code reuse
- Events for logging and monitoring
- Complex data structures (structs, mappings)
- Interface implementations

### DeFi Patterns
- Reflection mechanisms for token rewards
- Fee distribution systems
- Liquidity pool integrations
- Token burning mechanisms

### Governance Patterns
- Voting systems with deadlines
- Proposal creation and execution
- Multi-party authorization

## Learning Objectives

These contracts are designed to teach:

1. **Basic Solidity Syntax** - Variables, functions, modifiers, events
2. **Smart Contract Architecture** - Separation of concerns, modularity
3. **Security Best Practices** - Access control, input validation, reentrancy protection
4. **Gas Optimization** - Efficient coding patterns, storage optimization
5. **Token Standards** - ERC-20, ERC-721 implementations
6. **Advanced Patterns** - Multi-sig, governance, encryption, reflection rewards

## Usage for ML Training

These contracts provide diverse examples of:
- Different coding styles and patterns
- Various levels of complexity
- Real-world smart contract functionality
- Security considerations and implementations
- Gas optimization techniques

Perfect for training AI coding agents to understand and generate secure, efficient Solidity code.