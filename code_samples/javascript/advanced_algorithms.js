/**
 * Advanced Algorithms - Production Ready Implementations
 * Comprehensive algorithm implementations in JavaScript
 * 
 * Status: PRODUCTION READY
 * Last Updated: 2024
 * JavaScript Version: ES6+
 * Runtime: Node.js 14+ / Modern Browsers
 */

/**
 * Sorting Algorithms
 */
class SortingAlgorithms {
    /**
     * Quick Sort - Divide and conquer sorting
     * Time: O(n log n) average, O(n²) worst
     * Space: O(log n)
     */
    static quickSort(arr, low = 0, high = arr.length - 1) {
        if (low < high) {
            const pi = this.partition(arr, low, high);
            this.quickSort(arr, low, pi - 1);
            this.quickSort(arr, pi + 1, high);
        }
        return arr;
    }
    
    static partition(arr, low, high) {
        const pivot = arr[high];
        let i = low - 1;
        
        for (let j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                [arr[i], arr[j]] = [arr[j], arr[i]];
            }
        }
        
        [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
        return i + 1;
    }
    
    /**
     * Merge Sort - Stable divide and conquer
     * Time: O(n log n) all cases
     * Space: O(n)
     */
    static mergeSort(arr) {
        if (arr.length <= 1) return arr;
        
        const mid = Math.floor(arr.length / 2);
        const left = this.mergeSort(arr.slice(0, mid));
        const right = this.mergeSort(arr.slice(mid));
        
        return this.merge(left, right);
    }
    
    static merge(left, right) {
        const result = [];
        let i = 0, j = 0;
        
        while (i < left.length && j < right.length) {
            if (left[i] <= right[j]) {
                result.push(left[i++]);
            } else {
                result.push(right[j++]);
            }
        }
        
        return result.concat(left.slice(i)).concat(right.slice(j));
    }
    
    /**
     * Heap Sort - In-place sorting
     * Time: O(n log n) all cases
     * Space: O(1)
     */
    static heapSort(arr) {
        const n = arr.length;
        
        // Build max heap
        for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
            this.heapify(arr, n, i);
        }
        
        // Extract elements from heap
        for (let i = n - 1; i > 0; i--) {
            [arr[0], arr[i]] = [arr[i], arr[0]];
            this.heapify(arr, i, 0);
        }
        
        return arr;
    }
    
    static heapify(arr, n, i) {
        let largest = i;
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }
        
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }
        
        if (largest !== i) {
            [arr[i], arr[largest]] = [arr[largest], arr[i]];
            this.heapify(arr, n, largest);
        }
    }
}

/**
 * Searching Algorithms
 */
class SearchingAlgorithms {
    /**
     * Binary Search - Efficient search in sorted arrays
     * Time: O(log n)
     * Space: O(1)
     */
    static binarySearch(arr, target) {
        let left = 0, right = arr.length - 1;
        
        while (left <= right) {
            const mid = left + Math.floor((right - left) / 2);
            
            if (arr[mid] === target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }
    
    /**
     * Interpolation Search - Better for uniformly distributed data
     * Time: O(log log n) average, O(n) worst
     * Space: O(1)
     */
    static interpolationSearch(arr, target) {
        let left = 0, right = arr.length - 1;
        
        while (left <= right && target >= arr[left] && target <= arr[right]) {
            if (left === right) {
                return arr[left] === target ? left : -1;
            }
            
            const pos = left + Math.floor(
                ((target - arr[left]) * (right - left)) / 
                (arr[right] - arr[left])
            );
            
            if (arr[pos] === target) {
                return pos;
            } else if (arr[pos] < target) {
                left = pos + 1;
            } else {
                right = pos - 1;
            }
        }
        
        return -1;
    }
}

/**
 * Graph Algorithms
 */
class GraphAlgorithms {
    /**
     * Dijkstra's Shortest Path Algorithm
     * Time: O((V + E) log V) with binary heap
     * Space: O(V)
     */
    static dijkstra(graph, start) {
        const distances = {};
        const visited = new Set();
        const pq = [[0, start]]; // [distance, node]
        const previous = {};
        
        // Initialize distances
        for (const node in graph) {
            distances[node] = Infinity;
        }
        distances[start] = 0;
        
        while (pq.length > 0) {
            // Get node with minimum distance
            pq.sort((a, b) => a[0] - b[0]);
            const [currentDist, current] = pq.shift();
            
            if (visited.has(current)) continue;
            visited.add(current);
            
            // Update distances to neighbors
            for (const [neighbor, weight] of graph[current]) {
                const distance = currentDist + weight;
                
                if (distance < distances[neighbor]) {
                    distances[neighbor] = distance;
                    previous[neighbor] = current;
                    pq.push([distance, neighbor]);
                }
            }
        }
        
        return { distances, previous };
    }
    
    /**
     * Breadth-First Search
     * Time: O(V + E)
     * Space: O(V)
     */
    static bfs(graph, start) {
        const visited = new Set([start]);
        const queue = [start];
        const result = [];
        
        while (queue.length > 0) {
            const node = queue.shift();
            result.push(node);
            
            for (const neighbor of graph[node] || []) {
                const neighborNode = Array.isArray(neighbor) ? neighbor[0] : neighbor;
                if (!visited.has(neighborNode)) {
                    visited.add(neighborNode);
                    queue.push(neighborNode);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Depth-First Search
     * Time: O(V + E)
     * Space: O(V)
     */
    static dfs(graph, start, visited = new Set(), result = []) {
        visited.add(start);
        result.push(start);
        
        for (const neighbor of graph[start] || []) {
            const neighborNode = Array.isArray(neighbor) ? neighbor[0] : neighbor;
            if (!visited.has(neighborNode)) {
                this.dfs(graph, neighborNode, visited, result);
            }
        }
        
        return result;
    }
}

/**
 * Dynamic Programming Algorithms
 */
class DynamicProgramming {
    /**
     * 0/1 Knapsack Problem
     * Time: O(nW)
     * Space: O(nW)
     */
    static knapsack(weights, values, capacity) {
        const n = weights.length;
        const dp = Array(n + 1).fill(0).map(() => Array(capacity + 1).fill(0));
        
        for (let i = 1; i <= n; i++) {
            for (let w = 0; w <= capacity; w++) {
                if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(
                        dp[i - 1][w],
                        dp[i - 1][w - weights[i - 1]] + values[i - 1]
                    );
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        
        return dp[n][capacity];
    }
    
    /**
     * Longest Common Subsequence
     * Time: O(mn)
     * Space: O(mn)
     */
    static lcs(text1, text2) {
        const m = text1.length, n = text2.length;
        const dp = Array(m + 1).fill(0).map(() => Array(n + 1).fill(0));
        
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (text1[i - 1] === text2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }
    
    /**
     * Edit Distance (Levenshtein Distance)
     * Time: O(mn)
     * Space: O(mn)
     */
    static editDistance(word1, word2) {
        const m = word1.length, n = word2.length;
        const dp = Array(m + 1).fill(0).map(() => Array(n + 1).fill(0));
        
        // Initialize base cases
        for (let i = 0; i <= m; i++) dp[i][0] = i;
        for (let j = 0; j <= n; j++) dp[0][j] = j;
        
        // Fill DP table
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (word1[i - 1] === word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(
                        dp[i - 1][j],      // delete
                        dp[i][j - 1],      // insert
                        dp[i - 1][j - 1]   // replace
                    );
                }
            }
        }
        
        return dp[m][n];
    }
    
    /**
     * Coin Change - Minimum coins needed
     * Time: O(n * amount)
     * Space: O(amount)
     */
    static coinChange(coins, amount) {
        const dp = Array(amount + 1).fill(Infinity);
        dp[0] = 0;
        
        for (const coin of coins) {
            for (let i = coin; i <= amount; i++) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
        
        return dp[amount] === Infinity ? -1 : dp[amount];
    }
}

/**
 * String Algorithms
 */
class StringAlgorithms {
    /**
     * KMP (Knuth-Morris-Pratt) Pattern Matching
     * Time: O(n + m)
     * Space: O(m)
     */
    static kmpSearch(text, pattern) {
        if (!pattern) return [];
        
        const lps = this.buildLPS(pattern);
        const results = [];
        let i = 0, j = 0;
        
        while (i < text.length) {
            if (text[i] === pattern[j]) {
                i++;
                j++;
                
                if (j === pattern.length) {
                    results.push(i - j);
                    j = lps[j - 1];
                }
            } else {
                if (j !== 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }
        }
        
        return results;
    }
    
    static buildLPS(pattern) {
        const lps = Array(pattern.length).fill(0);
        let length = 0;
        let i = 1;
        
        while (i < pattern.length) {
            if (pattern[i] === pattern[length]) {
                length++;
                lps[i] = length;
                i++;
            } else {
                if (length !== 0) {
                    length = lps[length - 1];
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        
        return lps;
    }
    
    /**
     * Rabin-Karp Pattern Matching
     * Time: O(n + m) average, O(nm) worst
     * Space: O(1)
     */
    static rabinKarp(text, pattern, d = 256, q = 101) {
        const m = pattern.length;
        const n = text.length;
        const results = [];
        
        if (m > n) return results;
        
        let h = Math.pow(d, m - 1) % q;
        let p = 0, t = 0;
        
        // Calculate hash for pattern and first window
        for (let i = 0; i < m; i++) {
            p = (d * p + pattern.charCodeAt(i)) % q;
            t = (d * t + text.charCodeAt(i)) % q;
        }
        
        // Slide pattern over text
        for (let i = 0; i <= n - m; i++) {
            if (p === t) {
                // Hash match - verify actual string
                if (text.substr(i, m) === pattern) {
                    results.push(i);
                }
            }
            
            if (i < n - m) {
                t = (d * (t - text.charCodeAt(i) * h) + text.charCodeAt(i + m)) % q;
                if (t < 0) t += q;
            }
        }
        
        return results;
    }
}

// Demonstration and testing
if (typeof module !== 'undefined' && module.exports) {
    function runDemonstration() {
        console.log('='.repeat(60));
        console.log('Advanced Algorithms - Production Ready Examples');
        console.log('='.repeat(60));
        
        // Example 1: Sorting
        console.log('\n1. Sorting Algorithms');
        const unsorted = [64, 34, 25, 12, 22, 11, 90];
        console.log(`   Original: [${unsorted}]`);
        console.log(`   Quick Sort: [${SortingAlgorithms.quickSort([...unsorted])}]`);
        console.log(`   Merge Sort: [${SortingAlgorithms.mergeSort([...unsorted])}]`);
        console.log(`   Heap Sort: [${SortingAlgorithms.heapSort([...unsorted])}]`);
        
        // Example 2: Searching
        console.log('\n2. Searching Algorithms');
        const sorted = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
        const target = 11;
        console.log(`   Array: [${sorted}]`);
        console.log(`   Binary Search for ${target}: Index ${SearchingAlgorithms.binarySearch(sorted, target)}`);
        console.log(`   Interpolation Search for ${target}: Index ${SearchingAlgorithms.interpolationSearch(sorted, target)}`);
        
        // Example 3: Graph Algorithms
        console.log('\n3. Graph Algorithms');
        const graph = {
            'A': [['B', 4], ['C', 2]],
            'B': [['D', 5]],
            'C': [['B', 1], ['D', 8]],
            'D': []
        };
        console.log('   Graph: A->B(4), A->C(2), C->B(1), B->D(5), C->D(8)');
        const { distances } = GraphAlgorithms.dijkstra(graph, 'A');
        console.log('   Dijkstra from A:', distances);
        
        // Example 4: Dynamic Programming
        console.log('\n4. Dynamic Programming');
        const weights = [2, 3, 4, 5];
        const values = [3, 4, 5, 6];
        const capacity = 8;
        console.log(`   Knapsack: weights=[${weights}], values=[${values}], capacity=${capacity}`);
        console.log(`   Maximum value: ${DynamicProgramming.knapsack(weights, values, capacity)}`);
        
        const text1 = 'ABCDGH';
        const text2 = 'AEDFHR';
        console.log(`   LCS of "${text1}" and "${text2}": ${DynamicProgramming.lcs(text1, text2)}`);
        
        // Example 5: String Algorithms
        console.log('\n5. String Algorithms');
        const text = 'ABABDABACDABABCABAB';
        const pattern = 'ABABCABAB';
        console.log(`   Text: "${text}"`);
        console.log(`   Pattern: "${pattern}"`);
        console.log(`   KMP Search: Found at indices ${StringAlgorithms.kmpSearch(text, pattern)}`);
        
        console.log('\n' + '='.repeat(60));
        console.log('All examples completed successfully!');
        console.log('='.repeat(60));
    }
    
    // Run if executed directly
    if (require.main === module) {
        runDemonstration();
    }
    
    // Export modules
    module.exports = {
        SortingAlgorithms,
        SearchingAlgorithms,
        GraphAlgorithms,
        DynamicProgramming,
        StringAlgorithms
    };
}
