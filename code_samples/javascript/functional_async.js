// Sample JavaScript code for AI training dataset
// Demonstrates async/await patterns and functional programming

/**
 * Fetch data from an API with error handling
 * @param {string} url - The API endpoint
 * @returns {Promise<Object>} - The response data
 */
async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Fetch error:', error);
        throw error;
    }
}

/**
 * Array utility functions demonstrating functional programming
 */
const ArrayUtils = {
    // Filter even numbers
    filterEven: (arr) => arr.filter(num => num % 2 === 0),
    
    // Map to squares
    mapToSquares: (arr) => arr.map(num => num * num),
    
    // Reduce to sum
    sumAll: (arr) => arr.reduce((sum, num) => sum + num, 0),
    
    // Compose multiple operations
    processNumbers: (arr) => {
        return arr
            .filter(num => num > 0)
            .map(num => num * 2)
            .reduce((sum, num) => sum + num, 0);
    }
};

/**
 * Simple class demonstrating OOP in JavaScript
 */
class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
    
    multiply(a, b) {
        const result = a * b;
        this.history.push(`${a} * ${b} = ${result}`);
        return result;
    }
    
    getHistory() {
        return this.history.slice(); // Return copy
    }
    
    clearHistory() {
        this.history = [];
    }
}

// Example usage
const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const evenNumbers = ArrayUtils.filterEven(numbers);
const squares = ArrayUtils.mapToSquares(evenNumbers);
const sum = ArrayUtils.sumAll(squares);

console.log('Original numbers:', numbers);
console.log('Even numbers:', evenNumbers);
console.log('Squares:', squares);
console.log('Sum of squares:', sum);

const calc = new Calculator();
calc.add(5, 3);
calc.multiply(4, 7);
console.log('Calculator history:', calc.getHistory());