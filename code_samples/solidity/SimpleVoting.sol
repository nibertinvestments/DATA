// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

/**
 * @title SimpleVoting
 * @dev Basic voting contract with proposal creation and voting functionality
 */
contract SimpleVoting {
    struct Proposal {
        string description;
        uint256 voteCount;
        uint256 deadline;
        bool executed;
        mapping(address => bool) hasVoted;
    }
    
    mapping(uint256 => Proposal) public proposals;
    mapping(address => bool) public isVoter;
    
    uint256 public proposalCount;
    address public owner;
    uint256 public votingPeriod = 7 days;
    
    event ProposalCreated(uint256 indexed proposalId, string description, uint256 deadline);
    event VoteCast(uint256 indexed proposalId, address indexed voter);
    event ProposalExecuted(uint256 indexed proposalId);
    event VoterAdded(address indexed voter);
    event VoterRemoved(address indexed voter);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }
    
    modifier onlyVoter() {
        require(isVoter[msg.sender], "Not authorized to vote");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        isVoter[msg.sender] = true;
    }
    
    function addVoter(address _voter) external onlyOwner {
        require(_voter != address(0), "Invalid voter address");
        require(!isVoter[_voter], "Already a voter");
        
        isVoter[_voter] = true;
        emit VoterAdded(_voter);
    }
    
    function removeVoter(address _voter) external onlyOwner {
        require(isVoter[_voter], "Not a voter");
        
        isVoter[_voter] = false;
        emit VoterRemoved(_voter);
    }
    
    function createProposal(string memory _description) external onlyVoter {
        require(bytes(_description).length > 0, "Description cannot be empty");
        
        uint256 proposalId = proposalCount++;
        Proposal storage proposal = proposals[proposalId];
        proposal.description = _description;
        proposal.deadline = block.timestamp + votingPeriod;
        
        emit ProposalCreated(proposalId, _description, proposal.deadline);
    }
    
    function vote(uint256 _proposalId) external onlyVoter {
        Proposal storage proposal = proposals[_proposalId];
        require(block.timestamp <= proposal.deadline, "Voting period has ended");
        require(!proposal.hasVoted[msg.sender], "Already voted");
        require(!proposal.executed, "Proposal already executed");
        
        proposal.hasVoted[msg.sender] = true;
        proposal.voteCount++;
        
        emit VoteCast(_proposalId, msg.sender);
    }
    
    function executeProposal(uint256 _proposalId) external onlyOwner {
        Proposal storage proposal = proposals[_proposalId];
        require(block.timestamp > proposal.deadline, "Voting period not ended");
        require(!proposal.executed, "Proposal already executed");
        require(proposal.voteCount > 0, "No votes cast");
        
        proposal.executed = true;
        emit ProposalExecuted(_proposalId);
    }
    
    function getProposal(uint256 _proposalId) external view returns (
        string memory description,
        uint256 voteCount,
        uint256 deadline,
        bool executed
    ) {
        Proposal storage proposal = proposals[_proposalId];
        return (proposal.description, proposal.voteCount, proposal.deadline, proposal.executed);
    }
    
    function hasVoted(uint256 _proposalId, address _voter) external view returns (bool) {
        return proposals[_proposalId].hasVoted[_voter];
    }
    
    function setVotingPeriod(uint256 _newPeriod) external onlyOwner {
        require(_newPeriod > 0, "Voting period must be positive");
        votingPeriod = _newPeriod;
    }
}