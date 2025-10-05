/**
 * Vector Processing Operations - Production Ready
 * Advanced vector operations for ML/AI applications in JavaScript/Node.js
 * 
 * Status: PRODUCTION READY
 * Last Updated: 2024
 * JavaScript Version: ES6+
 * Runtime: Node.js 14+ / Modern Browsers
 */

/**
 * Core vector operations for ML/AI applications
 */
class VectorOperations {
    /**
     * Calculate cosine similarity between two vectors
     * @param {Array<number>} vec1 - First vector
     * @param {Array<number>} vec2 - Second vector
     * @returns {number} Cosine similarity score [-1, 1]
     */
    static cosineSimilarity(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have same dimensions');
        }

        const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
        const norm1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
        const norm2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));

        if (norm1 === 0 || norm2 === 0) {
            return 0;
        }

        return dotProduct / (norm1 * norm2);
    }

    /**
     * Calculate Euclidean distance between two vectors
     * @param {Array<number>} vec1 - First vector
     * @param {Array<number>} vec2 - Second vector
     * @returns {number} Euclidean distance (L2 norm)
     */
    static euclideanDistance(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have same dimensions');
        }

        const sumSquares = vec1.reduce((sum, val, i) => {
            const diff = val - vec2[i];
            return sum + diff * diff;
        }, 0);

        return Math.sqrt(sumSquares);
    }

    /**
     * Calculate Manhattan distance between two vectors
     * @param {Array<number>} vec1 - First vector
     * @param {Array<number>} vec2 - Second vector
     * @returns {number} Manhattan distance (L1 norm)
     */
    static manhattanDistance(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have same dimensions');
        }

        return vec1.reduce((sum, val, i) => sum + Math.abs(val - vec2[i]), 0);
    }

    /**
     * Normalize vector using L2 norm
     * @param {Array<number>} vector - Input vector
     * @returns {Array<number>} Normalized vector
     */
    static normalizeL2(vector) {
        const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
        
        if (norm === 0) {
            return [...vector];
        }

        return vector.map(val => val / norm);
    }

    /**
     * Dot product of two vectors
     * @param {Array<number>} vec1 - First vector
     * @param {Array<number>} vec2 - Second vector
     * @returns {number} Dot product
     */
    static dotProduct(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have same dimensions');
        }

        return vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
    }

    /**
     * Calculate batch cosine similarities (optimized)
     * @param {Array<Array<number>>} vectors - Matrix of vectors
     * @param {Array<number>} query - Query vector
     * @returns {Array<number>} Similarity scores
     */
    static batchCosineSimilarity(vectors, query) {
        const queryNorm = this.normalizeL2(query);
        
        return vectors.map(vec => {
            const vecNorm = this.normalizeL2(vec);
            return this.dotProduct(vecNorm, queryNorm);
        });
    }

    /**
     * Element-wise vector addition
     * @param {Array<number>} vec1 - First vector
     * @param {Array<number>} vec2 - Second vector
     * @returns {Array<number>} Sum vector
     */
    static add(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have same dimensions');
        }

        return vec1.map((val, i) => val + vec2[i]);
    }

    /**
     * Element-wise vector subtraction
     * @param {Array<number>} vec1 - First vector
     * @param {Array<number>} vec2 - Second vector
     * @returns {Array<number>} Difference vector
     */
    static subtract(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have same dimensions');
        }

        return vec1.map((val, i) => val - vec2[i]);
    }

    /**
     * Scalar multiplication
     * @param {Array<number>} vector - Input vector
     * @param {number} scalar - Scalar value
     * @returns {Array<number>} Scaled vector
     */
    static scale(vector, scalar) {
        return vector.map(val => val * scalar);
    }
}

/**
 * Vector embedding utilities
 */
class VectorEmbeddings {
    /**
     * Create simple word embedding (for demonstration)
     * In production, use pre-trained embeddings (Word2Vec, GloVe, etc.)
     * @param {string} word - Input word
     * @param {number} dim - Embedding dimension
     * @returns {Array<number>} Word embedding vector
     */
    static createWordEmbedding(word, dim = 300) {
        // Simple hash-based embedding for demonstration
        const seed = this._hashString(word.toLowerCase());
        const random = this._seededRandom(seed);
        
        const embedding = Array.from({ length: dim }, () => random() * 2 - 1);
        return VectorOperations.normalizeL2(embedding);
    }

    /**
     * Average pooling of multiple embeddings
     * @param {Array<Array<number>>} embeddings - List of embeddings
     * @returns {Array<number>} Averaged embedding
     */
    static averagePooling(embeddings) {
        if (embeddings.length === 0) {
            throw new Error('Empty embeddings list');
        }

        const dim = embeddings[0].length;
        const sum = new Array(dim).fill(0);

        embeddings.forEach(emb => {
            emb.forEach((val, i) => {
                sum[i] += val;
            });
        });

        return sum.map(val => val / embeddings.length);
    }

    /**
     * Max pooling of multiple embeddings
     * @param {Array<Array<number>>} embeddings - List of embeddings
     * @returns {Array<number>} Max-pooled embedding
     */
    static maxPooling(embeddings) {
        if (embeddings.length === 0) {
            throw new Error('Empty embeddings list');
        }

        const dim = embeddings[0].length;
        const result = new Array(dim).fill(-Infinity);

        embeddings.forEach(emb => {
            emb.forEach((val, i) => {
                result[i] = Math.max(result[i], val);
            });
        });

        return result;
    }

    /**
     * Simple hash function for strings
     * @private
     */
    static _hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return Math.abs(hash);
    }

    /**
     * Seeded random number generator
     * @private
     */
    static _seededRandom(seed) {
        let state = seed;
        return () => {
            state = (state * 1103515245 + 12345) & 0x7fffffff;
            return state / 0x7fffffff;
        };
    }
}

/**
 * In-memory vector database for similarity search
 */
class VectorDatabase {
    /**
     * Initialize vector database
     * @param {number} dim - Dimensionality of vectors
     */
    constructor(dim) {
        this.dim = dim;
        this.vectors = [];
        this.metadata = [];
    }

    /**
     * Add vector to database
     * @param {Array<number>} vector - Vector to add
     * @param {Object} metadata - Optional metadata
     */
    add(vector, metadata = {}) {
        if (vector.length !== this.dim) {
            throw new Error(`Vector dimension ${vector.length} doesn't match database dimension ${this.dim}`);
        }

        this.vectors.push([...vector]);
        this.metadata.push({ ...metadata });
    }

    /**
     * Add batch of vectors
     * @param {Array<Array<number>>} vectors - Vectors to add
     * @param {Array<Object>} metadataList - Optional metadata list
     */
    addBatch(vectors, metadataList = []) {
        vectors.forEach((vec, i) => {
            const meta = metadataList[i] || {};
            this.add(vec, meta);
        });
    }

    /**
     * Search for k most similar vectors
     * @param {Array<number>} query - Query vector
     * @param {number} k - Number of results
     * @param {string} metric - Similarity metric ('cosine', 'euclidean', 'manhattan')
     * @returns {Array<Object>} Results with index, score, and metadata
     */
    search(query, k = 10, metric = 'cosine') {
        if (this.vectors.length === 0) {
            return [];
        }

        let scores;
        let sortAscending = false;

        if (metric === 'cosine') {
            scores = VectorOperations.batchCosineSimilarity(this.vectors, query);
        } else if (metric === 'euclidean') {
            scores = this.vectors.map(vec => VectorOperations.euclideanDistance(vec, query));
            sortAscending = true;
        } else if (metric === 'manhattan') {
            scores = this.vectors.map(vec => VectorOperations.manhattanDistance(vec, query));
            sortAscending = true;
        } else {
            throw new Error(`Unknown similarity metric: ${metric}`);
        }

        // Create index-score pairs and sort
        const indexedScores = scores.map((score, idx) => ({ idx, score }));
        indexedScores.sort((a, b) => sortAscending ? a.score - b.score : b.score - a.score);

        // Return top k results
        return indexedScores.slice(0, k).map(({ idx, score }) => ({
            index: idx,
            score: score,
            metadata: this.metadata[idx]
        }));
    }

    /**
     * Get database size
     * @returns {number} Number of vectors
     */
    size() {
        return this.vectors.length;
    }

    /**
     * Clear database
     */
    clear() {
        this.vectors = [];
        this.metadata = [];
    }
}

/**
 * Vector quantization for compression
 */
class VectorQuantization {
    /**
     * Scalar quantization of vector
     * @param {Array<number>} vector - Input vector
     * @param {number} bits - Number of bits (8 or 16)
     * @returns {Object} Quantized data with codes, min, and scale
     */
    static scalarQuantization(vector, bits = 8) {
        const minVal = Math.min(...vector);
        const maxVal = Math.max(...vector);
        
        const nLevels = (2 ** bits) - 1;
        const scale = maxVal !== minVal ? (maxVal - minVal) / nLevels : 1.0;
        
        const quantized = vector.map(val => 
            Math.round((val - minVal) / scale)
        );
        
        return {
            quantized,
            minVal,
            scale,
            bits
        };
    }

    /**
     * Dequantize vector
     * @param {Array<number>} quantized - Quantized vector
     * @param {number} minVal - Minimum value from quantization
     * @param {number} scale - Scale from quantization
     * @returns {Array<number>} Dequantized vector
     */
    static dequantize(quantized, minVal, scale) {
        return quantized.map(val => val * scale + minVal);
    }

    /**
     * Calculate compression ratio
     * @param {number} originalBits - Original bits per value (32 or 64)
     * @param {number} quantizedBits - Quantized bits per value
     * @returns {number} Compression ratio
     */
    static compressionRatio(originalBits, quantizedBits) {
        return originalBits / quantizedBits;
    }
}

// Demonstration and testing
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    function runDemonstration() {
        console.log('='.repeat(60));
        console.log('Vector Processing Operations - Production Ready Examples');
        console.log('='.repeat(60));

        // Example 1: Cosine Similarity
        console.log('\n1. Cosine Similarity');
        const v1 = [1, 2, 3];
        const v2 = [4, 5, 6];
        const similarity = VectorOperations.cosineSimilarity(v1, v2);
        console.log(`   Vector 1: [${v1}]`);
        console.log(`   Vector 2: [${v2}]`);
        console.log(`   Cosine Similarity: ${similarity.toFixed(4)}`);

        // Example 2: Distance Metrics
        console.log('\n2. Distance Metrics');
        const euclidean = VectorOperations.euclideanDistance(v1, v2);
        const manhattan = VectorOperations.manhattanDistance(v1, v2);
        console.log(`   Euclidean Distance: ${euclidean.toFixed(4)}`);
        console.log(`   Manhattan Distance: ${manhattan.toFixed(4)}`);

        // Example 3: Vector Database
        console.log('\n3. Vector Database (Similarity Search)');
        const db = new VectorDatabase(128);
        
        // Add sample vectors
        for (let i = 0; i < 100; i++) {
            const vec = Array.from({ length: 128 }, () => Math.random() * 2 - 1);
            const normalized = VectorOperations.normalizeL2(vec);
            db.add(normalized, { id: i, category: `cat_${i % 5}` });
        }

        // Search
        const query = VectorOperations.normalizeL2(
            Array.from({ length: 128 }, () => Math.random() * 2 - 1)
        );
        const results = db.search(query, 5);

        console.log(`   Database size: ${db.size()} vectors`);
        console.log(`   Top 5 similar vectors:`);
        results.forEach(({ index, score, metadata }) => {
            console.log(`      Index ${index}: Score ${score.toFixed(4)}, Category: ${metadata.category}`);
        });

        // Example 4: Vector Operations
        console.log('\n4. Vector Operations');
        const a = [1, 2, 3];
        const b = [4, 5, 6];
        const sum = VectorOperations.add(a, b);
        const diff = VectorOperations.subtract(a, b);
        const scaled = VectorOperations.scale(a, 2);
        console.log(`   a + b = [${sum}]`);
        console.log(`   a - b = [${diff}]`);
        console.log(`   2 * a = [${scaled}]`);

        // Example 5: Quantization
        console.log('\n5. Vector Quantization (Compression)');
        const originalVec = Array.from({ length: 512 }, () => Math.random() * 10 - 5);
        const { quantized, minVal, scale } = VectorQuantization.scalarQuantization(originalVec, 8);
        const dequantized = VectorQuantization.dequantize(quantized, minVal, scale);
        
        const mse = originalVec.reduce((sum, val, i) => 
            sum + Math.pow(val - dequantized[i], 2), 0) / originalVec.length;
        
        console.log(`   Original size: ${originalVec.length * 4} bytes (float32)`);
        console.log(`   Quantized size: ${quantized.length * 1} bytes (uint8)`);
        console.log(`   Compression ratio: ${VectorQuantization.compressionRatio(32, 8)}x`);
        console.log(`   Reconstruction error (MSE): ${mse.toFixed(6)}`);

        // Example 6: Embeddings
        console.log('\n6. Word Embeddings');
        const words = ['hello', 'world', 'machine', 'learning'];
        const embeddings = words.map(word => VectorEmbeddings.createWordEmbedding(word, 300));
        const sentenceEmb = VectorEmbeddings.averagePooling(embeddings);
        console.log(`   Word embeddings dimension: ${embeddings[0].length}`);
        console.log(`   Sentence embedding (avg pool): [${sentenceEmb.slice(0, 5).map(v => v.toFixed(4))}...]`);

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
        VectorOperations,
        VectorEmbeddings,
        VectorDatabase,
        VectorQuantization
    };
}
