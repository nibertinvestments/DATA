// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

/**
 * @title SimpleNFT
 * @dev Basic ERC-721 NFT implementation with minting and metadata
 */
contract SimpleNFT {
    string public name;
    string public symbol;
    uint256 public totalSupply;
    
    mapping(uint256 => address) public ownerOf;
    mapping(address => uint256) public balanceOf;
    mapping(uint256 => address) public getApproved;
    mapping(address => mapping(address => bool)) public isApprovedForAll;
    mapping(uint256 => string) public tokenURI;
    
    address public owner;
    
    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);
    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }
    
    modifier tokenExists(uint256 _tokenId) {
        require(ownerOf[_tokenId] != address(0), "Token does not exist");
        _;
    }
    
    constructor(string memory _name, string memory _symbol) {
        name = _name;
        symbol = _symbol;
        owner = msg.sender;
    }
    
    function mint(address _to, uint256 _tokenId, string memory _tokenURI) external onlyOwner {
        require(_to != address(0), "Cannot mint to zero address");
        require(ownerOf[_tokenId] == address(0), "Token already minted");
        
        ownerOf[_tokenId] = _to;
        balanceOf[_to]++;
        totalSupply++;
        tokenURI[_tokenId] = _tokenURI;
        
        emit Transfer(address(0), _to, _tokenId);
    }
    
    function burn(uint256 _tokenId) external tokenExists(_tokenId) {
        address owner = ownerOf[_tokenId];
        require(msg.sender == owner || getApproved[_tokenId] == msg.sender || isApprovedForAll[owner][msg.sender], "Not authorized");
        
        delete ownerOf[_tokenId];
        delete getApproved[_tokenId];
        delete tokenURI[_tokenId];
        balanceOf[owner]--;
        totalSupply--;
        
        emit Transfer(owner, address(0), _tokenId);
    }
    
    function transfer(address _to, uint256 _tokenId) external {
        require(msg.sender == ownerOf[_tokenId], "Not the owner");
        _transfer(msg.sender, _to, _tokenId);
    }
    
    function transferFrom(address _from, address _to, uint256 _tokenId) external tokenExists(_tokenId) {
        require(_from == ownerOf[_tokenId], "From is not owner");
        require(
            msg.sender == _from ||
            getApproved[_tokenId] == msg.sender ||
            isApprovedForAll[_from][msg.sender],
            "Not authorized"
        );
        
        _transfer(_from, _to, _tokenId);
    }
    
    function approve(address _approved, uint256 _tokenId) external tokenExists(_tokenId) {
        address tokenOwner = ownerOf[_tokenId];
        require(msg.sender == tokenOwner || isApprovedForAll[tokenOwner][msg.sender], "Not authorized");
        
        getApproved[_tokenId] = _approved;
        emit Approval(tokenOwner, _approved, _tokenId);
    }
    
    function setApprovalForAll(address _operator, bool _approved) external {
        require(_operator != msg.sender, "Cannot approve yourself");
        isApprovedForAll[msg.sender][_operator] = _approved;
        emit ApprovalForAll(msg.sender, _operator, _approved);
    }
    
    function safeTransferFrom(address _from, address _to, uint256 _tokenId) external {
        transferFrom(_from, _to, _tokenId);
        require(_checkOnERC721Received(_from, _to, _tokenId, ""), "Transfer to non ERC721Receiver");
    }
    
    function safeTransferFrom(address _from, address _to, uint256 _tokenId, bytes memory _data) external {
        transferFrom(_from, _to, _tokenId);
        require(_checkOnERC721Received(_from, _to, _tokenId, _data), "Transfer to non ERC721Receiver");
    }
    
    function _transfer(address _from, address _to, uint256 _tokenId) internal {
        require(_to != address(0), "Cannot transfer to zero address");
        
        delete getApproved[_tokenId];
        balanceOf[_from]--;
        balanceOf[_to]++;
        ownerOf[_tokenId] = _to;
        
        emit Transfer(_from, _to, _tokenId);
    }
    
    function _checkOnERC721Received(address _from, address _to, uint256 _tokenId, bytes memory _data) 
        internal 
        returns (bool) 
    {
        if (_to.code.length > 0) {
            try IERC721Receiver(_to).onERC721Received(msg.sender, _from, _tokenId, _data) returns (bytes4 retval) {
                return retval == IERC721Receiver.onERC721Received.selector;
            } catch (bytes memory reason) {
                if (reason.length == 0) {
                    revert("Transfer to non ERC721Receiver");
                } else {
                    assembly {
                        revert(add(32, reason), mload(reason))
                    }
                }
            }
        } else {
            return true;
        }
    }
    
    function supportsInterface(bytes4 _interfaceId) external pure returns (bool) {
        return _interfaceId == 0x80ac58cd || // ERC721
               _interfaceId == 0x5b5e139f || // ERC721Metadata
               _interfaceId == 0x01ffc9a7;   // ERC165
    }
}

interface IERC721Receiver {
    function onERC721Received(address operator, address from, uint256 tokenId, bytes calldata data) external returns (bytes4);
}