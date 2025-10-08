// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * Design Patterns: Factory
 * AI/ML Training Sample
 */

contract Factory {
    string private data;
    
    constructor() {
        data = "";
    }
    
    function process(string memory input) public {
        data = input;
    }
    
    function getData() public view returns (string memory) {
        return data;
    }
    
    function validate() public view returns (bool) {
        return bytes(data).length > 0;
    }
}
