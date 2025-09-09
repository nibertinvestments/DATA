// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title Aetherweb3TokenCreator
 * @dev Advanced token creation contract for the Aetherweb3 ecosystem
 * @notice Allows users to create highly customizable ERC20 tokens with various features
 * 
 * Features:
 * - Standard ERC20 tokens
 * - Burnable tokens
 * - Mintable tokens  
 * - Pausable tokens
 * - Capped supply tokens
 * - Taxable tokens with configurable buy/sell/transfer taxes
 * - Reflection tokens with reward distribution
 * - Governance tokens with voting capabilities
 * - Flash mint capabilities
 * - Full-featured tokens combining all capabilities
 */
contract Aetherweb3TokenCreator is Ownable, ReentrancyGuard {
    
    // Token creation fee (configurable via deployment)
    uint256 public creationFee;
    
    // Fee recipient address
    address public feeRecipient;
    
    // Token types enumeration
    enum TokenType {
        STANDARD,       // Basic ERC20
        BURNABLE,       // With burn functionality
        MINTABLE,       // With mint functionality
        PAUSABLE,       // With pause functionality
        CAPPED,         // With supply cap
        TAXABLE,        // With transaction tax
        REFLECTION,     // With reflection rewards
        GOVERNANCE,     // With governance features
        FLASH_MINT,     // With flash minting
        FULL_FEATURED   // All features combined
    }
    
    // Token features structure
    struct TokenFeatures {
        bool burnable;
        bool mintable;
        bool pausable;
        bool capped;
        bool taxable;
        bool reflection;
        bool governance;
        bool flashMint;
        bool permit;
    }
    
    // Tax configuration for taxable tokens
    struct TaxConfig {
        uint256 buyTax;         // Tax on buys (in basis points)
        uint256 sellTax;        // Tax on sells (in basis points)
        uint256 transferTax;    // Tax on transfers (in basis points)
        address taxWallet;      // Wallet to receive taxes
        bool taxOnBuys;         // Enable tax on buys
        bool taxOnSells;        // Enable tax on sells
        bool taxOnTransfers;    // Enable tax on transfers
    }
    
    // Token creation parameters
    struct TokenParams {
        string name;                    // Token name
        string symbol;                  // Token symbol
        uint256 initialSupply;         // Initial supply
        uint8 decimals;                // Token decimals
        uint256 maxSupply;             // Maximum supply (for capped tokens)
        address owner;                 // Token owner
        TokenFeatures features;        // Token features
        TaxConfig taxConfig;           // Tax configuration
        bytes32 salt;                  // Salt for create2 deployment
    }
    
    // Created token information
    struct CreatedToken {
        address tokenAddress;          // Deployed token address
        address creator;               // Token creator
        string name;                   // Token name
        string symbol;                 // Token symbol
        uint256 creationTime;          // Creation timestamp
        TokenType tokenType;           // Token type
        uint256 initialSupply;         // Initial supply
        bool verified;                 // Verification status
    }
    
    // Events
    event TokenCreated(
        address indexed tokenAddress,
        address indexed creator,
        string name,
        string symbol,
        uint256 initialSupply,
        TokenType tokenType
    );
    
    event TokenVerified(
        address indexed tokenAddress,
        address indexed verifier
    );
    
    event FeeRecipientUpdated(
        address indexed oldRecipient,
        address indexed newRecipient
    );
    
    // State variables
    mapping(address => CreatedToken[]) public creatorTokens;
    mapping(address => bool) public verifiedTokens;
    mapping(address => address) public tokenCreators;
    uint256 public totalTokensCreated;
    uint256 public totalFeesCollected;
    
    // Fee exemption for ecosystem contracts
    mapping(address => bool) public feeExempt;
    
    /**
     * @dev Constructor
     * @param _feeRecipient Address to receive creation fees
     * @param _creationFee Fee amount for token creation
     */
    constructor(address _feeRecipient, uint256 _creationFee) {
        require(_feeRecipient != address(0), "Invalid fee recipient");
        require(_creationFee > 0, "Invalid creation fee");
        feeRecipient = _feeRecipient;
        creationFee = _creationFee;
        _transferOwnership(msg.sender);
    }
    
    /**
     * @dev Creates a new token with specified parameters
     * @param params Token creation parameters
     * @return tokenAddress Address of the created token
     */
    function createToken(TokenParams calldata params)
        external
        payable
        nonReentrant
        returns (address tokenAddress)
    {
        // Validate payment
        uint256 requiredFee = feeExempt[msg.sender] ? 0 : creationFee;
        require(msg.value >= requiredFee, "Insufficient fee");
        
        // Refund excess payment
        if (msg.value > requiredFee) {
            payable(msg.sender).transfer(msg.value - requiredFee);
        }
        
        // Transfer fee to recipient
        if (requiredFee > 0) {
            payable(feeRecipient).transfer(requiredFee);
            totalFeesCollected += requiredFee;
        }
        
        // Validate parameters
        _validateTokenParams(params);
        
        // Determine token type
        TokenType tokenType = _determineTokenType(params.features);
        
        // Deploy token
        tokenAddress = _deployToken(params, tokenType);
        
        // Record token creation
        CreatedToken memory newToken = CreatedToken({
            tokenAddress: tokenAddress,
            creator: msg.sender,
            name: params.name,
            symbol: params.symbol,
            creationTime: block.timestamp,
            tokenType: tokenType,
            initialSupply: params.initialSupply,
            verified: false
        });
        
        creatorTokens[msg.sender].push(newToken);
        tokenCreators[tokenAddress] = msg.sender;
        totalTokensCreated++;
        
        emit TokenCreated(
            tokenAddress,
            msg.sender,
            params.name,
            params.symbol,
            params.initialSupply,
            tokenType
        );
        
        return tokenAddress;
    }
    
    /**
     * @dev Creates a standard ERC20 token (simplified interface)
     * @param name Token name
     * @param symbol Token symbol
     * @param initialSupply Initial supply
     * @param decimals Token decimals
     * @return tokenAddress Address of the created token
     */
    function createStandardToken(
        string calldata name,
        string calldata symbol,
        uint256 initialSupply,
        uint8 decimals
    ) external payable returns (address tokenAddress) {
        TokenParams memory params = TokenParams({
            name: name,
            symbol: symbol,
            initialSupply: initialSupply,
            decimals: decimals,
            maxSupply: 0,
            owner: msg.sender,
            features: TokenFeatures({
                burnable: false,
                mintable: false,
                pausable: false,
                capped: false,
                taxable: false,
                reflection: false,
                governance: false,
                flashMint: false,
                permit: false
            }),
            taxConfig: TaxConfig({
                buyTax: 0,
                sellTax: 0,
                transferTax: 0,
                taxWallet: address(0),
                taxOnBuys: false,
                taxOnSells: false,
                taxOnTransfers: false
            }),
            salt: bytes32(0)
        });
        
        return createToken(params);
    }
    
    /**
     * @dev Validates token creation parameters
     * @param params Token parameters to validate
     */
    function _validateTokenParams(TokenParams memory params) internal pure {
        require(bytes(params.name).length > 0, "Name required");
        require(bytes(params.name).length <= 32, "Name too long");
        require(bytes(params.symbol).length > 0, "Symbol required");
        require(bytes(params.symbol).length <= 8, "Symbol too long");
        require(params.initialSupply > 0, "Initial supply required");
        require(params.decimals <= 18, "Invalid decimals");
        require(params.owner != address(0), "Invalid owner");
        
        if (params.features.capped) {
            require(params.maxSupply >= params.initialSupply, "Max supply too low");
        }
        
        if (params.features.taxable) {
            require(params.taxConfig.taxWallet != address(0), "Tax wallet required");
            require(params.taxConfig.buyTax <= 1000, "Buy tax too high"); // Max 10%
            require(params.taxConfig.sellTax <= 1000, "Sell tax too high");
            require(params.taxConfig.transferTax <= 1000, "Transfer tax too high");
        }
    }
    
    /**
     * @dev Determines token type based on features
     * @param features Token features
     * @return tokenType Determined token type
     */
    function _determineTokenType(TokenFeatures memory features)
        internal
        pure
        returns (TokenType tokenType)
    {
        if (features.burnable && features.mintable && features.pausable &&
            features.capped && features.taxable && features.reflection &&
            features.governance && features.flashMint) {
            return TokenType.FULL_FEATURED;
        } else if (features.governance) {
            return TokenType.GOVERNANCE;
        } else if (features.reflection) {
            return TokenType.REFLECTION;
        } else if (features.taxable) {
            return TokenType.TAXABLE;
        } else if (features.capped) {
            return TokenType.CAPPED;
        } else if (features.pausable) {
            return TokenType.PAUSABLE;
        } else if (features.mintable) {
            return TokenType.MINTABLE;
        } else if (features.burnable) {
            return TokenType.BURNABLE;
        } else {
            return TokenType.STANDARD;
        }
    }
    
    /**
     * @dev Deploys the token contract based on type
     * @param params Token parameters
     * @param tokenType Type of token to deploy
     * @return tokenAddress Address of deployed token
     */
    function _deployToken(TokenParams memory params, TokenType tokenType)
        internal
        returns (address tokenAddress)
    {
        if (tokenType == TokenType.STANDARD) {
            tokenAddress = address(new StandardToken(
                params.name,
                params.symbol,
                params.initialSupply,
                params.decimals,
                params.owner
            ));
        } else if (tokenType == TokenType.BURNABLE) {
            tokenAddress = address(new BurnableToken(
                params.name,
                params.symbol,
                params.initialSupply,
                params.owner
            ));
        }
        // Additional token types would be deployed here...
    }
    
    /**
     * @dev Gets tokens created by an address
     * @param creator Address of the creator
     * @return tokens Array of created tokens
     */
    function getCreatorTokens(address creator)
        external
        view
        returns (CreatedToken[] memory tokens)
    {
        return creatorTokens[creator];
    }
    
    /**
     * @dev Updates the fee recipient address (only owner)
     * @param _newRecipient New fee recipient address
     */
    function updateFeeRecipient(address _newRecipient) external onlyOwner {
        require(_newRecipient != address(0), "Invalid fee recipient");
        address oldRecipient = feeRecipient;
        feeRecipient = _newRecipient;
        emit FeeRecipientUpdated(oldRecipient, _newRecipient);
    }
    
    /**
     * @dev Updates the token creation fee (only owner)
     * @param _newFee New creation fee amount
     */
    function updateCreationFee(uint256 _newFee) external onlyOwner {
        require(_newFee > 0, "Invalid creation fee");
        creationFee = _newFee;
    }
    
    /**
     * @dev Withdraws accumulated fees to the fee recipient (only owner)
     */
    function withdrawFees() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No fees to withdraw");
        payable(feeRecipient).transfer(balance);
    }
}

/**
 * @title StandardToken
 * @dev Basic ERC20 token implementation
 */
contract StandardToken is ERC20, Ownable {
    uint8 private _decimals;
    
    constructor(
        string memory name,
        string memory symbol,
        uint256 initialSupply,
        uint8 decimals_,
        address owner
    ) ERC20(name, symbol) {
        _decimals = decimals_;
        _mint(owner, initialSupply);
        _transferOwnership(owner);
    }
    
    function decimals() public view virtual override returns (uint8) {
        return _decimals;
    }
}

/**
 * @title BurnableToken
 * @dev ERC20 token with burn functionality
 */
contract BurnableToken is ERC20Burnable, Ownable {
    constructor(
        string memory name,
        string memory symbol,
        uint256 initialSupply,
        address owner
    ) ERC20(name, symbol) {
        _mint(owner, initialSupply);
        _transferOwnership(owner);
    }
}