// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title Aetherweb3DAO
 * @dev Decentralized Autonomous Organization for governance
 * @notice This contract implements a comprehensive DAO system for decentralized governance
 * 
 * Features:
 * - Proposal creation and voting
 * - Time-delayed execution for security
 * - Quorum requirements for proposal validity
 * - Signature-based voting for gasless participation
 * - Multi-action proposals for complex governance
 */
contract Aetherweb3DAO is ReentrancyGuard, Ownable {
    
    // Governance token interface
    IERC20 public immutable governanceToken;
    
    // Timelock contract for delayed execution
    address public timelock;
    
    // Proposal structure containing all proposal data
    struct Proposal {
        uint256 id;                    // Unique proposal identifier
        address proposer;              // Address that created the proposal
        address[] targets;             // Target addresses for proposal calls
        uint256[] values;              // ETH values for proposal calls
        bytes[] calldatas;             // Calldata for proposal calls
        string description;            // Human-readable proposal description
        uint256 startTime;             // When voting starts
        uint256 endTime;               // When voting ends
        uint256 forVotes;              // Votes in favor
        uint256 againstVotes;          // Votes against
        uint256 abstainVotes;          // Abstain votes
        bool executed;                 // Whether proposal was executed
        bool canceled;                 // Whether proposal was canceled
        mapping(address => Receipt) receipts; // Individual vote receipts
    }
    
    // Vote receipt for tracking individual votes
    struct Receipt {
        bool hasVoted;                 // Whether the voter has voted
        uint8 support;                 // Vote type: 0=Against, 1=For, 2=Abstain
        uint256 votes;                 // Number of votes cast
    }
    
    // Possible states for a proposal
    enum ProposalState {
        Pending,    // Waiting for voting period to start
        Active,     // Currently accepting votes
        Canceled,   // Proposal was canceled
        Defeated,   // Proposal failed to meet requirements
        Succeeded,  // Proposal passed and ready for execution
        Queued,     // Proposal queued in timelock
        Expired,    // Proposal expired without execution
        Executed    // Proposal was successfully executed
    }
    
    // Governance parameters (constants for security)
    uint256 public constant VOTING_PERIOD = 7 days;           // 7 days voting period
    uint256 public constant VOTING_DELAY = 1 days;            // 1 day delay before voting starts
    uint256 public constant PROPOSAL_THRESHOLD = 100000 * 10**18; // 100k tokens to create proposal
    uint256 public constant QUORUM_PERCENTAGE = 10;           // 10% of total supply needed for quorum
    uint256 public constant MAX_OPERATIONS = 10;              // Maximum operations per proposal
    
    // State variables
    uint256 public proposalCount;                             // Total number of proposals created
    mapping(uint256 => Proposal) public proposals;            // Proposal storage
    mapping(address => uint256) public latestProposalIds;     // Latest proposal by each address
    
    // Events for transparency and off-chain monitoring
    event ProposalCreated(
        uint256 indexed proposalId,
        address indexed proposer,
        address[] targets,
        uint256[] values,
        bytes[] calldatas,
        string description,
        uint256 startTime,
        uint256 endTime
    );
    
    event VoteCast(
        address indexed voter,
        uint256 indexed proposalId,
        uint8 support,
        uint256 votes,
        string reason
    );
    
    event ProposalExecuted(uint256 indexed proposalId);
    event ProposalCanceled(uint256 indexed proposalId);
    event TimelockSet(address indexed oldTimelock, address indexed newTimelock);
    
    // Modifier to restrict functions to timelock only
    modifier onlyTimelock() {
        require(msg.sender == timelock, "DAO: caller must be timelock");
        _;
    }
    
    /**
     * @dev Constructor to initialize the DAO
     * @param _governanceToken Address of the governance token used for voting
     * @param _timelock Address of the timelock contract for delayed execution
     */
    constructor(address _governanceToken, address _timelock) {
        require(_governanceToken != address(0), "DAO: invalid governance token");
        require(_timelock != address(0), "DAO: invalid timelock");
        
        governanceToken = IERC20(_governanceToken);
        timelock = _timelock;
    }
    
    /**
     * @dev Create a new governance proposal
     * @param targets Target addresses for proposal calls
     * @param values ETH values for proposal calls
     * @param calldatas Calldata for proposal calls
     * @param description Human-readable proposal description
     * @return proposalId The ID of the created proposal
     */
    function propose(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description
    ) external returns (uint256) {
        // Validate proposer has enough tokens
        require(
            governanceToken.balanceOf(msg.sender) >= PROPOSAL_THRESHOLD,
            "DAO: proposer balance below threshold"
        );
        
        // Validate proposal structure
        require(targets.length == values.length, "DAO: invalid proposal length");
        require(targets.length == calldatas.length, "DAO: invalid proposal length");
        require(targets.length > 0, "DAO: empty proposal");
        require(targets.length <= MAX_OPERATIONS, "DAO: too many operations");
        require(bytes(description).length > 0, "DAO: empty description");
        
        // Prevent proposal spam from same address
        uint256 latestProposalId = latestProposalIds[msg.sender];
        if (latestProposalId != 0) {
            ProposalState proposalState = state(latestProposalId);
            require(
                proposalState != ProposalState.Active,
                "DAO: one live proposal per proposer"
            );
        }
        
        // Create new proposal
        uint256 proposalId = ++proposalCount;
        Proposal storage proposal = proposals[proposalId];
        
        proposal.id = proposalId;
        proposal.proposer = msg.sender;
        proposal.targets = targets;
        proposal.values = values;
        proposal.calldatas = calldatas;
        proposal.description = description;
        proposal.startTime = block.timestamp + VOTING_DELAY;
        proposal.endTime = proposal.startTime + VOTING_PERIOD;
        
        latestProposalIds[msg.sender] = proposalId;
        
        emit ProposalCreated(
            proposalId,
            msg.sender,
            targets,
            values,
            calldatas,
            description,
            proposal.startTime,
            proposal.endTime
        );
        
        return proposalId;
    }
    
    /**
     * @dev Cast a vote on a proposal
     * @param proposalId The proposal ID to vote on
     * @param support Vote type (0 = Against, 1 = For, 2 = Abstain)
     */
    function castVote(uint256 proposalId, uint8 support) external {
        _castVote(msg.sender, proposalId, support, "");
    }
    
    /**
     * @dev Cast a vote with a reason string
     * @param proposalId The proposal ID to vote on
     * @param support Vote type (0 = Against, 1 = For, 2 = Abstain)
     * @param reason Explanation for the vote
     */
    function castVoteWithReason(
        uint256 proposalId,
        uint8 support,
        string calldata reason
    ) external {
        _castVote(msg.sender, proposalId, support, reason);
    }
    
    /**
     * @dev Cast vote using signature (gasless voting)
     * @param proposalId The proposal ID
     * @param support Vote type
     * @param v Signature component v
     * @param r Signature component r
     * @param s Signature component s
     */
    function castVoteBySig(
        uint256 proposalId,
        uint8 support,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external {
        // Create EIP-712 domain separator
        bytes32 domainSeparator = keccak256(
            abi.encode(
                keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
                keccak256(bytes("Aetherweb3DAO")),
                keccak256(bytes("1")),
                block.chainid,
                address(this)
            )
        );
        
        // Create vote message hash
        bytes32 structHash = keccak256(
            abi.encode(
                keccak256("CastVote(uint256 proposalId,uint8 support)"),
                proposalId,
                support
            )
        );
        
        bytes32 digest = keccak256(abi.encodePacked("\x19\x01", domainSeparator, structHash));
        
        // Recover signer from signature
        address signatory = ecrecover(digest, v, r, s);
        require(signatory != address(0), "DAO: invalid signature");
        
        _castVote(signatory, proposalId, support, "");
    }
    
    /**
     * @dev Internal function to process vote casting
     * @param voter Address of the voter
     * @param proposalId Proposal being voted on
     * @param support Vote type
     * @param reason Reason for the vote (can be empty)
     */
    function _castVote(
        address voter,
        uint256 proposalId,
        uint8 support,
        string memory reason
    ) internal {
        require(state(proposalId) == ProposalState.Active, "DAO: proposal not active");
        require(support <= 2, "DAO: invalid vote type");
        
        Proposal storage proposal = proposals[proposalId];
        Receipt storage receipt = proposal.receipts[voter];
        
        require(!receipt.hasVoted, "DAO: voter already voted");
        
        uint256 votes = governanceToken.balanceOf(voter);
        require(votes > 0, "DAO: no voting power");
        
        // Record the vote
        receipt.hasVoted = true;
        receipt.support = support;
        receipt.votes = votes;
        
        // Update vote tallies
        if (support == 0) {
            proposal.againstVotes += votes;
        } else if (support == 1) {
            proposal.forVotes += votes;
        } else {
            proposal.abstainVotes += votes;
        }
        
        emit VoteCast(voter, proposalId, support, votes, reason);
    }
    
    /**
     * @dev Execute a successful proposal
     * @param proposalId The proposal ID to execute
     */
    function execute(uint256 proposalId) external payable nonReentrant {
        require(state(proposalId) == ProposalState.Succeeded, "DAO: proposal not successful");
        
        Proposal storage proposal = proposals[proposalId];
        proposal.executed = true;
        
        // Execute all operations in the proposal
        for (uint256 i = 0; i < proposal.targets.length; i++) {
            (bool success, ) = proposal.targets[i].call{value: proposal.values[i]}(
                proposal.calldatas[i]
            );
            require(success, "DAO: execution failed");
        }
        
        emit ProposalExecuted(proposalId);
    }
    
    /**
     * @dev Cancel a proposal (only proposer or high-stake holder)
     * @param proposalId The proposal ID to cancel
     */
    function cancel(uint256 proposalId) external {
        require(state(proposalId) != ProposalState.Executed, "DAO: cannot cancel executed proposal");
        
        Proposal storage proposal = proposals[proposalId];
        
        // Allow proposer to cancel or anyone with high stake
        require(
            msg.sender == proposal.proposer ||
            governanceToken.balanceOf(msg.sender) >= PROPOSAL_THRESHOLD,
            "DAO: insufficient rights to cancel"
        );
        
        proposal.canceled = true;
        emit ProposalCanceled(proposalId);
    }
    
    /**
     * @dev Get the current state of a proposal
     * @param proposalId The proposal ID
     * @return The current proposal state
     */
    function state(uint256 proposalId) public view returns (ProposalState) {
        require(proposalId <= proposalCount && proposalId > 0, "DAO: invalid proposal id");
        
        Proposal storage proposal = proposals[proposalId];
        
        if (proposal.canceled) {
            return ProposalState.Canceled;
        }
        
        if (proposal.executed) {
            return ProposalState.Executed;
        }
        
        if (block.timestamp <= proposal.startTime) {
            return ProposalState.Pending;
        }
        
        if (block.timestamp <= proposal.endTime) {
            return ProposalState.Active;
        }
        
        // Check if proposal succeeded
        if (_quorumReached(proposalId) && proposal.forVotes > proposal.againstVotes) {
            return ProposalState.Succeeded;
        }
        
        return ProposalState.Defeated;
    }
    
    /**
     * @dev Check if quorum requirement is met for a proposal
     * @param proposalId The proposal ID to check
     * @return True if quorum is reached
     */
    function _quorumReached(uint256 proposalId) internal view returns (bool) {
        Proposal storage proposal = proposals[proposalId];
        uint256 totalVotes = proposal.forVotes + proposal.againstVotes + proposal.abstainVotes;
        uint256 totalSupply = governanceToken.totalSupply();
        
        return totalVotes >= (totalSupply * QUORUM_PERCENTAGE) / 100;
    }
    
    /**
     * @dev Get comprehensive proposal information
     * @param proposalId The proposal ID
     * @return Tuple containing all proposal data
     */
    function getProposal(uint256 proposalId) external view returns (
        uint256 id,
        address proposer,
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description,
        uint256 startTime,
        uint256 endTime,
        uint256 forVotes,
        uint256 againstVotes,
        uint256 abstainVotes,
        bool executed,
        bool canceled
    ) {
        Proposal storage proposal = proposals[proposalId];
        return (
            proposal.id,
            proposal.proposer,
            proposal.targets,
            proposal.values,
            proposal.calldatas,
            proposal.description,
            proposal.startTime,
            proposal.endTime,
            proposal.forVotes,
            proposal.againstVotes,
            proposal.abstainVotes,
            proposal.executed,
            proposal.canceled
        );
    }
    
    /**
     * @dev Get vote receipt for a specific voter on a proposal
     * @param proposalId The proposal ID
     * @param voter The voter address
     * @return Vote receipt information
     */
    function getReceipt(uint256 proposalId, address voter) external view returns (
        bool hasVoted,
        uint8 support,
        uint256 votes
    ) {
        Receipt storage receipt = proposals[proposalId].receipts[voter];
        return (receipt.hasVoted, receipt.support, receipt.votes);
    }
    
    /**
     * @dev Update timelock address (only owner)
     * @param newTimelock New timelock contract address
     */
    function setTimelock(address newTimelock) external onlyOwner {
        require(newTimelock != address(0), "DAO: invalid timelock");
        address oldTimelock = timelock;
        timelock = newTimelock;
        emit TimelockSet(oldTimelock, newTimelock);
    }
    
    /**
     * @dev Get total number of proposals created
     * @return Total proposal count
     */
    function getProposalCount() external view returns (uint256) {
        return proposalCount;
    }
    
    /**
     * @dev Check if an address can create proposals
     * @param account Address to check
     * @return True if account meets proposal threshold
     */
    function canPropose(address account) external view returns (bool) {
        return governanceToken.balanceOf(account) >= PROPOSAL_THRESHOLD;
    }
    
    /**
     * @dev Get voting power for an address
     * @param account Address to check
     * @return Voting power (token balance)
     */
    function getVotingPower(address account) external view returns (uint256) {
        return governanceToken.balanceOf(account);
    }
    
    /**
     * @dev Receive function to accept ETH donations
     */
    receive() external payable {
        // DAO can receive ETH for treasury operations
    }
}