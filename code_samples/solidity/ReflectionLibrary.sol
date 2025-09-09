// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

/**
 * @title ReflectionRewards
 * @dev A Solidity library for implementing reflection rewards with configurable fees for creator, reflection, liquidity pool, and burn.
 *      Designed for ERC-20 token contracts. Includes exclusion from fees/reflection, batch operations, explicit reflection claiming, and secure, gas-optimized logic.
 *      All internal functions are used by the calling contract.
 */
library ReflectionRewards {
    // Constants
    uint256 private constant _MAX_FEE_BPS = 10000; // 100% in basis points
    uint256 private constant _PRECISION = 1e18; // Precision for reflection calculations, multiply before divide to minimize loss
    uint256 private constant _MAX_UINT256 = type(uint256).max;
    uint256 private constant _MAX_BATCH_SIZE = 100; // Maximum batch size to prevent gas limit issues

    /**
     * @dev Struct to hold fee configuration for a token.
     */
    struct _FeeConfig {
        uint256 creatorFeeBps; // Fee for creator in basis points (1% = 100 bps)
        uint256 reflectionFeeBps; // Fee for reflection in basis points
        uint256 liquidityFeeBps; // Fee for liquidity pool in basis points
        uint256 burnFeeBps; // Fee to burn tokens in basis points
        uint256 totalFeeBps; // Cached total fee (sum of all fees)
        address creatorAddress; // Address to receive creator fees
        address liquidityPool; // Address to receive liquidity pool fees
        bool isInitialized; // Tracks if config is initialized
    }

    /**
     * @dev Struct to store reflection-related data for an account.
     */
    struct _AccountData {
        uint256 reflectedBalance; // Balance including reflected tokens
        uint256 excludedBalance; // Balance excluded from reflection
        uint256 lastReflectionPerToken; // Last reflectionPerToken value when balance was updated
        bool isExcludedFromFees; // Whether the account is exempt from fees
        bool isExcludedFromReflection; // Whether the account is excluded from reflection
    }

    /**
     * @dev Struct to track reflection distribution state.
     */
    struct _ReflectionState {
        uint256 totalReflected; // Total tokens distributed as reflection
        uint256 totalExcludedSupply; // Total supply excluded from reflection
        uint256 reflectionPerToken; // Reflection amount per token (scaled by _PRECISION)
    }

    // Events
    event FeesUpdated(uint256 indexed creatorFeeBps, uint256 indexed reflectionFeeBps, uint256 liquidityFeeBps, uint256 burnFeeBps);
    event CreatorAddressUpdated(address indexed newCreator);
    event LiquidityPoolUpdated(address indexed newLiquidityPool);
    event AccountExcludedFromFees(address indexed account, bool isExcluded);
    event AccountExcludedFromReflection(address indexed account, bool isExcluded);
    event ReflectionDistributed(uint256 indexed amount);
    event FeeConfigInitialized(address indexed creatorAddress, address indexed liquidityPool);
    event TokensBurned(uint256 indexed amount);
    event ReflectionsClaimed(address indexed account, uint256 indexed amount);
    event BalanceUpdated(address indexed account, uint256 newBalance);

    /**
     * @dev Initializes the fee configuration. Called by the token contract.
     */
    function initialize(
        _FeeConfig storage config,
        uint256 creatorFeeBps,
        uint256 reflectionFeeBps,
        uint256 liquidityFeeBps,
        uint256 burnFeeBps,
        address creatorAddress,
        address liquidityPool
    ) internal {
        require(!config.isInitialized, "Already initialized");
        require(creatorAddress != address(0), "Invalid creator");
        require(liquidityPool != address(0), "Invalid liquidity");
        require(creatorFeeBps + reflectionFeeBps + liquidityFeeBps + burnFeeBps < _MAX_FEE_BPS, "Fees too high");

        config.creatorFeeBps = creatorFeeBps;
        config.reflectionFeeBps = reflectionFeeBps;
        config.liquidityFeeBps = liquidityFeeBps;
        config.burnFeeBps = burnFeeBps;
        config.totalFeeBps = creatorFeeBps + reflectionFeeBps + liquidityFeeBps + burnFeeBps;
        config.creatorAddress = creatorAddress;
        config.liquidityPool = liquidityPool;
        config.isInitialized = true;

        emit FeeConfigInitialized(creatorAddress, liquidityPool);
        emit FeesUpdated(creatorFeeBps, reflectionFeeBps, liquidityFeeBps, burnFeeBps);
        emit CreatorAddressUpdated(creatorAddress);
        emit LiquidityPoolUpdated(liquidityPool);
    }

    /**
     * @dev Updates the fee configuration post-deployment. Called by the token contract.
     */
    function updateFees(
        _FeeConfig storage config,
        uint256 creatorFeeBps,
        uint256 reflectionFeeBps,
        uint256 liquidityFeeBps,
        uint256 burnFeeBps
    ) internal {
        require(config.isInitialized, "Not initialized");
        require(creatorFeeBps + reflectionFeeBps + liquidityFeeBps + burnFeeBps < _MAX_FEE_BPS, "Fees too high");

        config.creatorFeeBps = creatorFeeBps;
        config.reflectionFeeBps = reflectionFeeBps;
        config.liquidityFeeBps = liquidityFeeBps;
        config.burnFeeBps = burnFeeBps;
        config.totalFeeBps = creatorFeeBps + reflectionFeeBps + liquidityFeeBps + burnFeeBps;

        emit FeesUpdated(creatorFeeBps, reflectionFeeBps, liquidityFeeBps, burnFeeBps);
    }

    /**
     * @dev Updates the creator address post-deployment. Called by the token contract.
     */
    function updateCreatorAddress(_FeeConfig storage config, address newCreatorAddress) internal {
        require(config.isInitialized, "Not initialized");
        require(newCreatorAddress != address(0), "Invalid creator");
        config.creatorAddress = newCreatorAddress;
        emit CreatorAddressUpdated(newCreatorAddress);
    }

    /**
     * @dev Updates the liquidity pool address post-deployment. Called by the token contract.
     */
    function updateLiquidityPool(_FeeConfig storage config, address newLiquidityPool) internal {
        require(config.isInitialized, "Not initialized");
        require(newLiquidityPool != address(0), "Invalid liquidity");
        config.liquidityPool = newLiquidityPool;
        emit LiquidityPoolUpdated(newLiquidityPool);
    }

    /**
     * @dev Excludes or includes an account from fees. Called by the token contract.
     */
    function setFeeExclusion(
        mapping(address => _AccountData) storage accounts,
        address account,
        bool isExcluded
    ) internal {
        require(account != address(0), "Invalid account");
        accounts[account].isExcludedFromFees = isExcluded;
        emit AccountExcludedFromFees(account, isExcluded);
    }

    /**
     * @dev Excludes or includes an account from reflection. Called by the token contract.
     */
    function setReflectionExclusion(
        mapping(address => _AccountData) storage accounts,
        _ReflectionState storage state,
        address account,
        bool isExcluded,
        uint256 totalSupply
    ) internal {
        require(account != address(0), "Invalid account");
        require(totalSupply != 0, "Invalid supply");
        _AccountData storage accountData = accounts[account];

        if (accountData.isExcludedFromReflection == isExcluded) {
            return;
        }

        updateReflectedBalance(accounts, state, account);

        if (isExcluded) {
            uint256 currentBalance = accountData.reflectedBalance;
            require(state.totalExcludedSupply + currentBalance < totalSupply, "Excluded supply overflow");
            accountData.excludedBalance = currentBalance;
            delete accountData.reflectedBalance;
            state.totalExcludedSupply += currentBalance;
        } else {
            uint256 excludedBalance = accountData.excludedBalance;
            require(state.totalExcludedSupply > excludedBalance, "Excluded supply underflow");
            accountData.reflectedBalance = excludedBalance;
            delete accountData.excludedBalance;
            state.totalExcludedSupply -= excludedBalance;
            accountData.lastReflectionPerToken = state.reflectionPerToken;
        }

        accountData.isExcludedFromReflection = isExcluded;
        emit AccountExcludedFromReflection(account, isExcluded);
    }

    /**
     * @dev Processes a transfer, applying fees and updating reflection balances.
     * @return amountReceived The amount received by the recipient after fees.
     */
    function processTransfer(
        _FeeConfig storage config,
        mapping(address => _AccountData) storage accounts,
        _ReflectionState storage state,
        address sender,
        address recipient,
        uint256 amount,
        uint256 totalSupply
    ) internal returns (uint256 amountReceived) {
        require(config.isInitialized, "Not initialized");
        require(amount != 0, "Invalid amount");
        require(sender != address(0), "Invalid sender");
        require(recipient != address(0), "Invalid recipient");
        require(totalSupply != 0, "Invalid supply");

        _AccountData storage senderData = accounts[sender];
        _AccountData storage recipientData = accounts[recipient];

        updateReflectedBalance(accounts, state, sender);
        updateReflectedBalance(accounts, state, recipient);

        uint256 senderBalance = getRealBalance(accounts, state, sender);
        require(senderBalance > amount, "Insufficient balance");

        bool isFeeExempt = senderData.isExcludedFromFees || recipientData.isExcludedFromFees;

        if (isFeeExempt || config.totalFeeBps == 0) {
            updateBalances(accounts, state, sender, recipient, amount, totalSupply);
            return amount;
        }

        // Calculate fees
        uint256 totalFee;
        uint256 creatorFee;
        uint256 reflectionFee;
        uint256 liquidityFee;
        uint256 burnFee;
        unchecked { // Safe: totalFeeBps < _MAX_FEE_BPS
            totalFee = (amount * config.totalFeeBps) / _MAX_FEE_BPS;
            creatorFee = (amount * config.creatorFeeBps) / _MAX_FEE_BPS;
            reflectionFee = (amount * config.reflectionFeeBps) / _MAX_FEE_BPS;
            liquidityFee = (amount * config.liquidityFeeBps) / _MAX_FEE_BPS;
            burnFee = (amount * config.burnFeeBps) / _MAX_FEE_BPS;
        }

        require(totalFee == creatorFee + reflectionFee + liquidityFee + burnFee, "Fee mismatch");
        require(amount > totalFee, "Amount too small");

        amountReceived = amount - totalFee;

        updateBalances(accounts, state, sender, recipient, amountReceived, totalSupply);

        if (creatorFee != 0) {
            updateReflectedBalance(accounts, state, config.creatorAddress);
            accounts[config.creatorAddress].reflectedBalance += creatorFee;
            emit BalanceUpdated(config.creatorAddress, accounts[config.creatorAddress].reflectedBalance);
        }
        if (liquidityFee != 0) {
            updateReflectedBalance(accounts, state, config.liquidityPool);
            accounts[config.liquidityPool].reflectedBalance += liquidityFee;
            emit BalanceUpdated(config.liquidityPool, accounts[config.liquidityPool].reflectedBalance);
        }
        if (reflectionFee != 0) {
            distributeReflection(state, reflectionFee, totalSupply);
        }
        if (burnFee != 0) {
            emit TokensBurned(burnFee);
        }
    }

    /**
     * @dev Processes batch transfers to multiple recipients.
     * @return amountsReceived Array of amounts received by each recipient after fees.
     */
    function processBatchTransfer(
        _FeeConfig storage config,
        mapping(address => _AccountData) storage accounts,
        _ReflectionState storage state,
        address sender,
        address[] memory recipients,
        uint256[] memory amounts,
        uint256 totalSupply
    ) internal returns (uint256[] memory amountsReceived) {
        require(config.isInitialized, "Not initialized");
        require(sender != address(0), "Invalid sender");
        require(recipients.length == amounts.length, "Arrays length mismatch");
        require(recipients.length != 0, "Empty recipients");
        require(recipients.length < _MAX_BATCH_SIZE, "Batch too large");
        require(totalSupply != 0, "Invalid supply");

        amountsReceived = new uint256[](recipients.length);

        updateReflectedBalance(accounts, state, sender);

        uint256 length = recipients.length;
        uint256 totalAmount;
        for (uint256 i = 0; i < length; ) {
            require(recipients[i] != address(0), "Invalid recipient");
            require(amounts[i] != 0, "Invalid amount");
            unchecked { totalAmount += amounts[i]; }
            unchecked { ++i; }
        }

        uint256 senderBalance = getRealBalance(accounts, state, sender);
        require(senderBalance > totalAmount, "Insufficient balance");

        for (uint256 i = 0; i < length; ) {
            amountsReceived[i] = processTransfer(config, accounts, state, sender, recipients[i], amounts[i], totalSupply);
            unchecked { ++i; }
        }
    }

    /**
     * @dev Updates balances for sender and recipient during a transfer. Called by processTransfer and processBatchTransfer.
     */
    function updateBalances(
        mapping(address => _AccountData) storage accounts,
        _ReflectionState storage state,
        address sender,
        address recipient,
        uint256 amount,
        uint256 totalSupply
    ) private {
        _AccountData storage senderData = accounts[sender];
        _AccountData storage recipientData = accounts[recipient];

        if (senderData.isExcludedFromReflection) {
            require(senderData.excludedBalance > amount, "Insufficient excluded balance");
            senderData.excludedBalance -= amount;
        } else {
            require(senderData.reflectedBalance > amount, "Insufficient reflected balance");
            senderData.reflectedBalance -= amount;
        }
        emit BalanceUpdated(sender, senderData.isExcludedFromReflection ? senderData.excludedBalance : senderData.reflectedBalance);

        if (recipientData.isExcludedFromReflection) {
            require(state.totalExcludedSupply + amount < totalSupply, "Excluded supply overflow");
            recipientData.excludedBalance += amount;
            state.totalExcludedSupply += amount;
        } else {
            recipientData.reflectedBalance += amount;
        }
        emit BalanceUpdated(recipient, recipientData.isExcludedFromReflection ? recipientData.excludedBalance : recipientData.reflectedBalance);
    }

    /**
     * @dev Distributes reflection fees to all eligible holders. Called by processTransfer.
     */
    function distributeReflection(
        _ReflectionState storage state,
        uint256 reflectionFee,
        uint256 totalSupply
    ) private {
        if (reflectionFee == 0) return;

        require(totalSupply > state.totalExcludedSupply, "Invalid excluded supply");
        uint256 reflectableSupply = totalSupply - state.totalExcludedSupply;
        if (reflectableSupply == 0) return;

        uint256 reflectionPerTokenDelta;
        unchecked { // Safe: reflectableSupply > 0
            reflectionPerTokenDelta = (reflectionFee * _PRECISION) / reflectableSupply;
        }
        require(_MAX_UINT256 - state.reflectionPerToken > reflectionPerTokenDelta, "Reflection per token overflow");
        state.reflectionPerToken += reflectionPerTokenDelta;
        state.totalReflected += reflectionFee;

        emit ReflectionDistributed(reflectionFee);
    }

    /**
     * @dev Calculates the real balance (including reflection) for an account.
     */
    function getRealBalance(
        mapping(address => _AccountData) storage accounts,
        _ReflectionState storage state,
        address account
    ) internal view returns (uint256 balance) {
        _AccountData storage accountData = accounts[account];
        if (accountData.isExcludedFromReflection) {
            return accountData.excludedBalance;
        }
        uint256 reflectionDelta;
        unchecked { // Safe: reflectionPerToken is cumulative
            reflectionDelta = state.reflectionPerToken - accountData.lastReflectionPerToken;
        }
        uint256 owedReflection;
        unchecked { // Safe: _PRECISION is large
            owedReflection = (accountData.reflectedBalance * reflectionDelta) / _PRECISION;
        }
        return accountData.reflectedBalance + owedReflection;
    }

    /**
     * @dev Updates an account's reflected balance to include accumulated reflections. Called by processTransfer and claimReflections.
     */
    function updateReflectedBalance(
        mapping(address => _AccountData) storage accounts,
        _ReflectionState storage state,
        address account
    ) internal {
        require(account != address(0), "Invalid account");
        _AccountData storage accountData = accounts[account];
        if (accountData.isExcludedFromReflection) return;

        uint256 reflectionDelta;
        unchecked { // Safe: reflectionPerToken is cumulative
            reflectionDelta = state.reflectionPerToken - accountData.lastReflectionPerToken;
        }
        uint256 owedReflection;
        unchecked { // Safe: _PRECISION is large
            owedReflection = (accountData.reflectedBalance * reflectionDelta) / _PRECISION;
        }
        require(_MAX_UINT256 - accountData.reflectedBalance > owedReflection, "Balance overflow");
        accountData.reflectedBalance += owedReflection;
        accountData.lastReflectionPerToken = state.reflectionPerToken;

        if (owedReflection != 0) {
            emit ReflectionsClaimed(account, owedReflection);
        }
    }

    /**
     * @dev Allows an account to explicitly claim accumulated reflections. Called by the token contract.
     */
    function claimReflections(
        mapping(address => _AccountData) storage accounts,
        _ReflectionState storage state,
        address account
    ) internal {
        require(account != address(0), "Invalid account");
        updateReflectedBalance(accounts, state, account);
    }

    /**
     * @dev Excludes multiple accounts from reflection in a single transaction. Called by the token contract.
     */
    function setBatchReflectionExclusion(
        mapping(address => _AccountData) storage accounts,
        _ReflectionState storage state,
        address[] memory addresses,
        bool isExcluded,
        uint256 totalSupply
    ) internal {
        require(addresses.length != 0, "Empty addresses");
        require(addresses.length < _MAX_BATCH_SIZE, "Batch too large");
        require(totalSupply != 0, "Invalid supply");

        uint256 length = addresses.length;
        for (uint256 i = 0; i < length; ) {
            require(addresses[i] != address(0), "Invalid address");
            setReflectionExclusion(accounts, state, addresses[i], isExcluded, totalSupply);
            unchecked { ++i; }
        }
    }

    /**
     * @dev Excludes multiple accounts from fees in a single transaction. Called by the token contract.
     */
    function setBatchFeeExclusion(
        mapping(address => _AccountData) storage accounts,
        address[] memory addresses,
        bool isExcluded
    ) internal {
        require(addresses.length != 0, "Empty addresses");
        require(addresses.length < _MAX_BATCH_SIZE, "Batch too large");

        uint256 length = addresses.length;
        for (uint256 i = 0; i < length; ) {
            require(addresses[i] != address(0), "Invalid address");
            setFeeExclusion(accounts, addresses[i], isExcluded);
            unchecked { ++i; }
        }
    }
}