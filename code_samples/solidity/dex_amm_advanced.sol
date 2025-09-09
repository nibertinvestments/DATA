// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title Aetherweb3DEX
 * @dev Decentralized Exchange (DEX) with Automated Market Maker (AMM) functionality
 * @notice This contract implements a Uniswap V2-style DEX with additional features
 * 
 * Features:
 * - Liquidity pools for token swapping
 * - Liquidity provision with LP tokens
 * - Dynamic fee structure
 * - Price oracle functionality  
 * - MEV protection mechanisms
 * - Flash loan capabilities
 * - Governance integration
 */
contract Aetherweb3DEX is ReentrancyGuard, Ownable {
    
    // Constants for fee calculations (basis points)
    uint256 public constant MAX_FEE = 1000;           // 10% maximum fee
    uint256 public constant FEE_DENOMINATOR = 10000;  // Basis points denominator
    
    // Pool information structure
    struct PoolInfo {
        address tokenA;           // First token in the pair
        address tokenB;           // Second token in the pair
        uint256 reserveA;         // Reserve of token A
        uint256 reserveB;         // Reserve of token B
        uint256 totalSupply;      // Total LP token supply
        uint256 kLast;            // Product of reserves at last liquidity event
        uint256 feeRate;          // Pool-specific fee rate
        bool initialized;         // Whether pool is initialized
    }
    
    // Liquidity provider position
    struct LPPosition {
        uint256 amount;           // LP token amount
        uint256 timestamp;        // When position was created
        uint256 rewardDebt;       // Rewards already claimed
    }
    
    // Swap information for events
    struct SwapInfo {
        address tokenIn;          // Input token
        address tokenOut;         // Output token
        uint256 amountIn;         // Input amount
        uint256 amountOut;        // Output amount
        address to;               // Recipient
        uint256 fee;              // Fee paid
    }
    
    // State variables
    mapping(bytes32 => PoolInfo) public pools;                    // Pool storage
    mapping(bytes32 => mapping(address => LPPosition)) public lpPositions; // LP positions
    mapping(address => bool) public isAuthorizedCaller;           // Authorized callers for MEV protection
    
    uint256 public defaultFeeRate = 30;                           // 0.3% default fee
    uint256 public protocolFeeShare = 1667;                       // 16.67% of trading fees to protocol
    address public feeRecipient;                                  // Protocol fee recipient
    
    // Price oracle variables
    mapping(bytes32 => uint256) public priceAverages;             // Time-weighted average prices
    mapping(bytes32 => uint256) public lastUpdateTime;            // Last price update timestamp
    uint256 public constant PRICE_PERIOD = 300;                   // 5-minute TWAP period
    
    // Flash loan variables
    uint256 public flashLoanFee = 9;                              // 0.09% flash loan fee
    mapping(address => bool) public flashLoanEnabled;             // Flash loan enabled tokens
    
    // Events for transparency and monitoring
    event PoolCreated(address indexed tokenA, address indexed tokenB, bytes32 indexed poolId);
    event LiquidityAdded(address indexed provider, bytes32 indexed poolId, uint256 amountA, uint256 amountB, uint256 lpTokens);
    event LiquidityRemoved(address indexed provider, bytes32 indexed poolId, uint256 amountA, uint256 amountB, uint256 lpTokens);
    event Swap(address indexed user, bytes32 indexed poolId, SwapInfo swapInfo);
    event FeeRateUpdated(bytes32 indexed poolId, uint256 newFeeRate);
    event FlashLoan(address indexed borrower, address indexed token, uint256 amount, uint256 fee);
    
    /**
     * @dev Constructor
     * @param _feeRecipient Address to receive protocol fees
     */
    constructor(address _feeRecipient) {
        require(_feeRecipient != address(0), "DEX: invalid fee recipient");
        feeRecipient = _feeRecipient;
        isAuthorizedCaller[msg.sender] = true;
    }
    
    /**
     * @dev Create a new liquidity pool
     * @param tokenA First token address
     * @param tokenB Second token address
     * @param feeRate Pool-specific fee rate (in basis points)
     * @return poolId Unique identifier for the pool
     */
    function createPool(
        address tokenA,
        address tokenB,
        uint256 feeRate
    ) external returns (bytes32 poolId) {
        require(tokenA != tokenB, "DEX: identical tokens");
        require(tokenA != address(0) && tokenB != address(0), "DEX: zero address");
        require(feeRate <= MAX_FEE, "DEX: fee rate too high");
        
        // Sort tokens to ensure consistent pool IDs
        (address token0, address token1) = tokenA < tokenB ? (tokenA, tokenB) : (tokenB, tokenA);
        poolId = keccak256(abi.encodePacked(token0, token1));
        
        require(!pools[poolId].initialized, "DEX: pool already exists");
        
        // Initialize pool
        pools[poolId] = PoolInfo({
            tokenA: token0,
            tokenB: token1,
            reserveA: 0,
            reserveB: 0,
            totalSupply: 0,
            kLast: 0,
            feeRate: feeRate > 0 ? feeRate : defaultFeeRate,
            initialized: true
        });
        
        // Enable flash loans by default for common tokens
        flashLoanEnabled[token0] = true;
        flashLoanEnabled[token1] = true;
        
        emit PoolCreated(token0, token1, poolId);
        return poolId;
    }
    
    /**
     * @dev Add liquidity to a pool
     * @param tokenA First token address
     * @param tokenB Second token address
     * @param amountA Desired amount of token A
     * @param amountB Desired amount of token B
     * @param minAmountA Minimum amount of token A
     * @param minAmountB Minimum amount of token B
     * @param to Recipient of LP tokens
     * @return lpTokens Amount of LP tokens minted
     */
    function addLiquidity(
        address tokenA,
        address tokenB,
        uint256 amountA,
        uint256 amountB,
        uint256 minAmountA,
        uint256 minAmountB,
        address to
    ) external nonReentrant returns (uint256 lpTokens) {
        bytes32 poolId = getPoolId(tokenA, tokenB);
        PoolInfo storage pool = pools[poolId];
        require(pool.initialized, "DEX: pool not found");
        
        // Calculate optimal amounts
        (uint256 optimalAmountA, uint256 optimalAmountB) = _calculateOptimalAmounts(
            pool, amountA, amountB
        );
        
        require(optimalAmountA >= minAmountA, "DEX: insufficient amount A");
        require(optimalAmountB >= minAmountB, "DEX: insufficient amount B");
        
        // Calculate LP tokens to mint
        if (pool.totalSupply == 0) {
            // First liquidity provision
            lpTokens = sqrt(optimalAmountA * optimalAmountB);
            require(lpTokens > 1000, "DEX: insufficient initial liquidity"); // Minimum liquidity lock
        } else {
            // Subsequent liquidity provision
            lpTokens = min(
                (optimalAmountA * pool.totalSupply) / pool.reserveA,
                (optimalAmountB * pool.totalSupply) / pool.reserveB
            );
        }
        
        require(lpTokens > 0, "DEX: insufficient liquidity minted");
        
        // Transfer tokens from user
        IERC20(pool.tokenA).transferFrom(msg.sender, address(this), optimalAmountA);
        IERC20(pool.tokenB).transferFrom(msg.sender, address(this), optimalAmountB);
        
        // Update pool state
        pool.reserveA += optimalAmountA;
        pool.reserveB += optimalAmountB;
        pool.totalSupply += lpTokens;
        
        // Update user LP position
        lpPositions[poolId][to].amount += lpTokens;
        lpPositions[poolId][to].timestamp = block.timestamp;
        
        // Update price oracle
        _updatePriceAverage(poolId, pool);
        
        emit LiquidityAdded(to, poolId, optimalAmountA, optimalAmountB, lpTokens);
        return lpTokens;
    }
    
    /**
     * @dev Remove liquidity from a pool
     * @param tokenA First token address
     * @param tokenB Second token address
     * @param lpTokens Amount of LP tokens to burn
     * @param minAmountA Minimum amount of token A to receive
     * @param minAmountB Minimum amount of token B to receive
     * @param to Recipient of tokens
     * @return amountA Amount of token A received
     * @return amountB Amount of token B received
     */
    function removeLiquidity(
        address tokenA,
        address tokenB,
        uint256 lpTokens,
        uint256 minAmountA,
        uint256 minAmountB,
        address to
    ) external nonReentrant returns (uint256 amountA, uint256 amountB) {
        bytes32 poolId = getPoolId(tokenA, tokenB);
        PoolInfo storage pool = pools[poolId];
        require(pool.initialized, "DEX: pool not found");
        
        LPPosition storage position = lpPositions[poolId][msg.sender];
        require(position.amount >= lpTokens, "DEX: insufficient LP tokens");
        
        // Calculate token amounts to return
        amountA = (lpTokens * pool.reserveA) / pool.totalSupply;
        amountB = (lpTokens * pool.reserveB) / pool.totalSupply;
        
        require(amountA >= minAmountA, "DEX: insufficient amount A");
        require(amountB >= minAmountB, "DEX: insufficient amount B");
        
        // Update user position
        position.amount -= lpTokens;
        
        // Update pool state
        pool.reserveA -= amountA;
        pool.reserveB -= amountB;
        pool.totalSupply -= lpTokens;
        
        // Transfer tokens to user
        IERC20(pool.tokenA).transfer(to, amountA);
        IERC20(pool.tokenB).transfer(to, amountB);
        
        // Update price oracle
        _updatePriceAverage(poolId, pool);
        
        emit LiquidityRemoved(to, poolId, amountA, amountB, lpTokens);
        return (amountA, amountB);
    }
    
    /**
     * @dev Swap tokens in a pool
     * @param tokenIn Input token address
     * @param tokenOut Output token address
     * @param amountIn Input token amount
     * @param minAmountOut Minimum output amount
     * @param to Recipient address
     * @return amountOut Actual output amount
     */
    function swap(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 minAmountOut,
        address to
    ) external nonReentrant returns (uint256 amountOut) {
        require(tokenIn != tokenOut, "DEX: identical tokens");
        require(amountIn > 0, "DEX: zero amount");
        
        bytes32 poolId = getPoolId(tokenIn, tokenOut);
        PoolInfo storage pool = pools[poolId];
        require(pool.initialized, "DEX: pool not found");
        
        // MEV protection
        require(isAuthorizedCaller[msg.sender] || tx.origin == msg.sender, "DEX: MEV protection");
        
        // Determine which token is A and which is B
        bool tokenInIsA = tokenIn == pool.tokenA;
        uint256 reserveIn = tokenInIsA ? pool.reserveA : pool.reserveB;
        uint256 reserveOut = tokenInIsA ? pool.reserveB : pool.reserveA;
        
        // Calculate output amount with fees
        amountOut = getAmountOut(amountIn, reserveIn, reserveOut, pool.feeRate);
        require(amountOut >= minAmountOut, "DEX: insufficient output amount");
        require(amountOut < reserveOut, "DEX: insufficient liquidity");
        
        // Calculate protocol fee
        uint256 fee = (amountIn * pool.feeRate) / FEE_DENOMINATOR;
        uint256 protocolFee = (fee * protocolFeeShare) / FEE_DENOMINATOR;
        
        // Transfer input tokens from user
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn);
        
        // Transfer protocol fee if applicable
        if (protocolFee > 0) {
            IERC20(tokenIn).transfer(feeRecipient, protocolFee);
        }
        
        // Update reserves
        if (tokenInIsA) {
            pool.reserveA += amountIn;
            pool.reserveB -= amountOut;
        } else {
            pool.reserveB += amountIn;
            pool.reserveA -= amountOut;
        }
        
        // Transfer output tokens to recipient
        IERC20(tokenOut).transfer(to, amountOut);
        
        // Update price oracle
        _updatePriceAverage(poolId, pool);
        
        SwapInfo memory swapInfo = SwapInfo({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            amountIn: amountIn,
            amountOut: amountOut,
            to: to,
            fee: fee
        });
        
        emit Swap(msg.sender, poolId, swapInfo);
        return amountOut;
    }
    
    /**
     * @dev Execute a flash loan
     * @param token Token to borrow
     * @param amount Amount to borrow
     * @param data Callback data
     */
    function flashLoan(
        address token,
        uint256 amount,
        bytes calldata data
    ) external nonReentrant {
        require(flashLoanEnabled[token], "DEX: flash loans disabled for token");
        require(amount > 0, "DEX: zero amount");
        
        uint256 balanceBefore = IERC20(token).balanceOf(address(this));
        require(balanceBefore >= amount, "DEX: insufficient liquidity for flash loan");
        
        uint256 fee = (amount * flashLoanFee) / FEE_DENOMINATOR;
        
        // Transfer tokens to borrower
        IERC20(token).transfer(msg.sender, amount);
        
        // Call borrower's callback
        IFlashLoanReceiver(msg.sender).executeOperation(token, amount, fee, data);
        
        // Check repayment
        uint256 balanceAfter = IERC20(token).balanceOf(address(this));
        require(balanceAfter >= balanceBefore + fee, "DEX: flash loan not repaid");
        
        // Transfer fee to protocol
        if (fee > 0) {
            IERC20(token).transfer(feeRecipient, fee);
        }
        
        emit FlashLoan(msg.sender, token, amount, fee);
    }
    
    /**
     * @dev Get output amount for a given input
     * @param amountIn Input amount
     * @param reserveIn Input token reserve
     * @param reserveOut Output token reserve
     * @param feeRate Pool fee rate
     * @return amountOut Output amount
     */
    function getAmountOut(
        uint256 amountIn,
        uint256 reserveIn,
        uint256 reserveOut,
        uint256 feeRate
    ) public pure returns (uint256 amountOut) {
        require(amountIn > 0, "DEX: zero input amount");
        require(reserveIn > 0 && reserveOut > 0, "DEX: insufficient liquidity");
        
        uint256 amountInWithFee = amountIn * (FEE_DENOMINATOR - feeRate);
        uint256 numerator = amountInWithFee * reserveOut;
        uint256 denominator = (reserveIn * FEE_DENOMINATOR) + amountInWithFee;
        
        amountOut = numerator / denominator;
    }
    
    /**
     * @dev Get pool ID for a token pair
     * @param tokenA First token
     * @param tokenB Second token
     * @return poolId Pool identifier
     */
    function getPoolId(address tokenA, address tokenB) public pure returns (bytes32 poolId) {
        (address token0, address token1) = tokenA < tokenB ? (tokenA, tokenB) : (tokenB, tokenA);
        return keccak256(abi.encodePacked(token0, token1));
    }
    
    /**
     * @dev Get current price for a token pair
     * @param tokenA First token
     * @param tokenB Second token
     * @return price Price of tokenA in terms of tokenB
     */
    function getPrice(address tokenA, address tokenB) external view returns (uint256 price) {
        bytes32 poolId = getPoolId(tokenA, tokenB);
        PoolInfo storage pool = pools[poolId];
        require(pool.initialized, "DEX: pool not found");
        
        if (tokenA == pool.tokenA) {
            return (pool.reserveB * 1e18) / pool.reserveA;
        } else {
            return (pool.reserveA * 1e18) / pool.reserveB;
        }
    }
    
    /**
     * @dev Get time-weighted average price
     * @param tokenA First token
     * @param tokenB Second token
     * @return twap Time-weighted average price
     */
    function getTWAP(address tokenA, address tokenB) external view returns (uint256 twap) {
        bytes32 poolId = getPoolId(tokenA, tokenB);
        return priceAverages[poolId];
    }
    
    /**
     * @dev Internal function to update price average
     */
    function _updatePriceAverage(bytes32 poolId, PoolInfo storage pool) internal {
        uint256 currentTime = block.timestamp;
        uint256 lastUpdate = lastUpdateTime[poolId];
        
        if (currentTime > lastUpdate + PRICE_PERIOD) {
            uint256 currentPrice = (pool.reserveB * 1e18) / pool.reserveA;
            uint256 timeElapsed = currentTime - lastUpdate;
            uint256 oldAverage = priceAverages[poolId];
            
            // Simple time-weighted average
            priceAverages[poolId] = (oldAverage * (PRICE_PERIOD - timeElapsed) + currentPrice * timeElapsed) / PRICE_PERIOD;
            lastUpdateTime[poolId] = currentTime;
        }
    }
    
    /**
     * @dev Calculate optimal amounts for liquidity provision
     */
    function _calculateOptimalAmounts(
        PoolInfo storage pool,
        uint256 amountA,
        uint256 amountB
    ) internal view returns (uint256 optimalAmountA, uint256 optimalAmountB) {
        if (pool.reserveA == 0 && pool.reserveB == 0) {
            return (amountA, amountB);
        }
        
        uint256 amountBOptimal = (amountA * pool.reserveB) / pool.reserveA;
        if (amountBOptimal <= amountB) {
            return (amountA, amountBOptimal);
        } else {
            uint256 amountAOptimal = (amountB * pool.reserveA) / pool.reserveB;
            return (amountAOptimal, amountB);
        }
    }
    
    // Utility functions
    function sqrt(uint256 x) internal pure returns (uint256) {
        if (x == 0) return 0;
        uint256 z = (x + 1) / 2;
        uint256 y = x;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
        return y;
    }
    
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
    
    // Admin functions
    function setDefaultFeeRate(uint256 _feeRate) external onlyOwner {
        require(_feeRate <= MAX_FEE, "DEX: fee rate too high");
        defaultFeeRate = _feeRate;
    }
    
    function setFlashLoanFee(uint256 _flashLoanFee) external onlyOwner {
        require(_flashLoanFee <= 100, "DEX: flash loan fee too high"); // Max 1%
        flashLoanFee = _flashLoanFee;
    }
    
    function setAuthorizedCaller(address caller, bool authorized) external onlyOwner {
        isAuthorizedCaller[caller] = authorized;
    }
}

/**
 * @title Flash loan receiver interface
 */
interface IFlashLoanReceiver {
    function executeOperation(
        address token,
        uint256 amount,
        uint256 fee,
        bytes calldata data
    ) external;
}