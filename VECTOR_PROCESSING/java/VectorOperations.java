/**
 * Vector Processing Operations - Production Ready
 * Advanced vector operations for ML/AI applications in Java
 * 
 * Status: PRODUCTION READY
 * Last Updated: 2024
 * Java Version: 11+
 * 
 * This implementation provides production-ready vector operations including:
 * - Similarity metrics (cosine, euclidean, manhattan)
 * - Vector database with similarity search
 * - Vector embeddings and pooling operations
 * - Vector quantization for compression
 */

package vectorprocessing;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Core vector operations for ML/AI applications
 */
public class VectorOperations {
    
    /**
     * Calculate cosine similarity between two vectors
     * @param vec1 First vector
     * @param vec2 Second vector
     * @return Cosine similarity score [-1, 1]
     * @throws IllegalArgumentException if vectors have different dimensions
     */
    public static double cosineSimilarity(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Vectors must have same dimensions");
        }
        
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);
        
        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }
        
        return dotProduct / (norm1 * norm2);
    }
    
    /**
     * Calculate Euclidean distance between two vectors
     * @param vec1 First vector
     * @param vec2 Second vector
     * @return Euclidean distance (L2 norm)
     */
    public static double euclideanDistance(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Vectors must have same dimensions");
        }
        
        double sumSquares = 0.0;
        for (int i = 0; i < vec1.length; i++) {
            double diff = vec1[i] - vec2[i];
            sumSquares += diff * diff;
        }
        
        return Math.sqrt(sumSquares);
    }
    
    /**
     * Calculate Manhattan distance between two vectors
     * @param vec1 First vector
     * @param vec2 Second vector
     * @return Manhattan distance (L1 norm)
     */
    public static double manhattanDistance(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Vectors must have same dimensions");
        }
        
        double sum = 0.0;
        for (int i = 0; i < vec1.length; i++) {
            sum += Math.abs(vec1[i] - vec2[i]);
        }
        
        return sum;
    }
    
    /**
     * Normalize vector using L2 norm
     * @param vector Input vector
     * @return Normalized vector
     */
    public static double[] normalizeL2(double[] vector) {
        double norm = 0.0;
        for (double v : vector) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        
        if (norm == 0.0) {
            return Arrays.copyOf(vector, vector.length);
        }
        
        double[] normalized = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        
        return normalized;
    }
    
    /**
     * Calculate dot product of two vectors
     * @param vec1 First vector
     * @param vec2 Second vector
     * @return Dot product
     */
    public static double dotProduct(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Vectors must have same dimensions");
        }
        
        double result = 0.0;
        for (int i = 0; i < vec1.length; i++) {
            result += vec1[i] * vec2[i];
        }
        
        return result;
    }
    
    /**
     * Calculate batch cosine similarities
     * @param vectors Matrix of vectors
     * @param query Query vector
     * @return Array of similarity scores
     */
    public static double[] batchCosineSimilarity(List<double[]> vectors, double[] query) {
        double[] queryNorm = normalizeL2(query);
        double[] similarities = new double[vectors.size()];
        
        for (int i = 0; i < vectors.size(); i++) {
            double[] vecNorm = normalizeL2(vectors.get(i));
            similarities[i] = dotProduct(vecNorm, queryNorm);
        }
        
        return similarities;
    }
    
    /**
     * Element-wise vector addition
     * @param vec1 First vector
     * @param vec2 Second vector
     * @return Sum vector
     */
    public static double[] add(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Vectors must have same dimensions");
        }
        
        double[] result = new double[vec1.length];
        for (int i = 0; i < vec1.length; i++) {
            result[i] = vec1[i] + vec2[i];
        }
        
        return result;
    }
    
    /**
     * Element-wise vector subtraction
     * @param vec1 First vector
     * @param vec2 Second vector
     * @return Difference vector
     */
    public static double[] subtract(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Vectors must have same dimensions");
        }
        
        double[] result = new double[vec1.length];
        for (int i = 0; i < vec1.length; i++) {
            result[i] = vec1[i] - vec2[i];
        }
        
        return result;
    }
    
    /**
     * Scalar multiplication
     * @param vector Input vector
     * @param scalar Scalar value
     * @return Scaled vector
     */
    public static double[] scale(double[] vector, double scalar) {
        double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i] * scalar;
        }
        
        return result;
    }
}

/**
 * Vector embedding utilities
 */
class VectorEmbeddings {
    
    /**
     * Create simple word embedding (for demonstration)
     * In production, use pre-trained embeddings (Word2Vec, GloVe, BERT)
     * @param word Input word
     * @param dim Embedding dimension
     * @return Word embedding vector
     */
    public static double[] createWordEmbedding(String word, int dim) {
        // Simple hash-based embedding for demonstration
        long seed = word.toLowerCase().hashCode();
        Random random = new Random(seed);
        
        double[] embedding = new double[dim];
        for (int i = 0; i < dim; i++) {
            embedding[i] = random.nextGaussian();
        }
        
        return VectorOperations.normalizeL2(embedding);
    }
    
    /**
     * Average pooling of multiple embeddings
     * @param embeddings List of embeddings
     * @return Averaged embedding
     */
    public static double[] averagePooling(List<double[]> embeddings) {
        if (embeddings.isEmpty()) {
            throw new IllegalArgumentException("Empty embeddings list");
        }
        
        int dim = embeddings.get(0).length;
        double[] sum = new double[dim];
        
        for (double[] emb : embeddings) {
            for (int i = 0; i < dim; i++) {
                sum[i] += emb[i];
            }
        }
        
        for (int i = 0; i < dim; i++) {
            sum[i] /= embeddings.size();
        }
        
        return sum;
    }
    
    /**
     * Max pooling of multiple embeddings
     * @param embeddings List of embeddings
     * @return Max-pooled embedding
     */
    public static double[] maxPooling(List<double[]> embeddings) {
        if (embeddings.isEmpty()) {
            throw new IllegalArgumentException("Empty embeddings list");
        }
        
        int dim = embeddings.get(0).length;
        double[] result = new double[dim];
        Arrays.fill(result, Double.NEGATIVE_INFINITY);
        
        for (double[] emb : embeddings) {
            for (int i = 0; i < dim; i++) {
                result[i] = Math.max(result[i], emb[i]);
            }
        }
        
        return result;
    }
}

/**
 * Search result container
 */
class SearchResult {
    public final int index;
    public final double score;
    public final Map<String, Object> metadata;
    
    public SearchResult(int index, double score, Map<String, Object> metadata) {
        this.index = index;
        this.score = score;
        this.metadata = metadata;
    }
    
    @Override
    public String toString() {
        return String.format("SearchResult{index=%d, score=%.4f, metadata=%s}", 
                           index, score, metadata);
    }
}

/**
 * In-memory vector database for similarity search
 */
class VectorDatabase {
    private final int dim;
    private final List<double[]> vectors;
    private final List<Map<String, Object>> metadata;
    
    /**
     * Initialize vector database
     * @param dim Dimensionality of vectors
     */
    public VectorDatabase(int dim) {
        this.dim = dim;
        this.vectors = new ArrayList<>();
        this.metadata = new ArrayList<>();
    }
    
    /**
     * Add vector to database
     * @param vector Vector to add
     * @param meta Optional metadata
     */
    public void add(double[] vector, Map<String, Object> meta) {
        if (vector.length != dim) {
            throw new IllegalArgumentException(
                String.format("Vector dimension %d doesn't match database dimension %d", 
                            vector.length, dim));
        }
        
        vectors.add(Arrays.copyOf(vector, vector.length));
        metadata.add(new HashMap<>(meta));
    }
    
    /**
     * Add vector with empty metadata
     * @param vector Vector to add
     */
    public void add(double[] vector) {
        add(vector, new HashMap<>());
    }
    
    /**
     * Add batch of vectors
     * @param vectors Vectors to add
     * @param metadataList Optional metadata list
     */
    public void addBatch(List<double[]> vectors, List<Map<String, Object>> metadataList) {
        for (int i = 0; i < vectors.size(); i++) {
            Map<String, Object> meta = (metadataList != null && i < metadataList.size()) 
                                       ? metadataList.get(i) 
                                       : new HashMap<>();
            add(vectors.get(i), meta);
        }
    }
    
    /**
     * Search for k most similar vectors
     * @param query Query vector
     * @param k Number of results
     * @param metric Similarity metric ('cosine', 'euclidean', 'manhattan')
     * @return List of search results
     */
    public List<SearchResult> search(double[] query, int k, String metric) {
        if (vectors.isEmpty()) {
            return new ArrayList<>();
        }
        
        List<SearchResult> results = new ArrayList<>();
        
        for (int i = 0; i < vectors.size(); i++) {
            double score;
            
            switch (metric.toLowerCase()) {
                case "cosine":
                    score = VectorOperations.cosineSimilarity(vectors.get(i), query);
                    break;
                case "euclidean":
                    score = VectorOperations.euclideanDistance(vectors.get(i), query);
                    break;
                case "manhattan":
                    score = VectorOperations.manhattanDistance(vectors.get(i), query);
                    break;
                default:
                    throw new IllegalArgumentException("Unknown similarity metric: " + metric);
            }
            
            results.add(new SearchResult(i, score, metadata.get(i)));
        }
        
        // Sort results
        boolean ascending = !metric.equalsIgnoreCase("cosine");
        results.sort((a, b) -> ascending ? 
                    Double.compare(a.score, b.score) : 
                    Double.compare(b.score, a.score));
        
        // Return top k
        return results.stream().limit(k).collect(Collectors.toList());
    }
    
    /**
     * Search with default cosine similarity
     * @param query Query vector
     * @param k Number of results
     * @return List of search results
     */
    public List<SearchResult> search(double[] query, int k) {
        return search(query, k, "cosine");
    }
    
    /**
     * Get database size
     * @return Number of vectors
     */
    public int size() {
        return vectors.size();
    }
    
    /**
     * Clear database
     */
    public void clear() {
        vectors.clear();
        metadata.clear();
    }
}

/**
 * Vector quantization for compression
 */
class VectorQuantization {
    
    /**
     * Result of scalar quantization
     */
    public static class QuantizationResult {
        public final byte[] quantized;
        public final double minVal;
        public final double scale;
        public final int bits;
        
        public QuantizationResult(byte[] quantized, double minVal, double scale, int bits) {
            this.quantized = quantized;
            this.minVal = minVal;
            this.scale = scale;
            this.bits = bits;
        }
    }
    
    /**
     * Scalar quantization of vector
     * @param vector Input vector
     * @param bits Number of bits (8)
     * @return Quantization result
     */
    public static QuantizationResult scalarQuantization(double[] vector, int bits) {
        double minVal = Arrays.stream(vector).min().orElse(0.0);
        double maxVal = Arrays.stream(vector).max().orElse(0.0);
        
        int nLevels = (1 << bits) - 1;
        double scale = (maxVal != minVal) ? (maxVal - minVal) / nLevels : 1.0;
        
        byte[] quantized = new byte[vector.length];
        for (int i = 0; i < vector.length; i++) {
            quantized[i] = (byte) Math.round((vector[i] - minVal) / scale);
        }
        
        return new QuantizationResult(quantized, minVal, scale, bits);
    }
    
    /**
     * Dequantize vector
     * @param result Quantization result
     * @return Dequantized vector
     */
    public static double[] dequantize(QuantizationResult result) {
        double[] dequantized = new double[result.quantized.length];
        
        for (int i = 0; i < result.quantized.length; i++) {
            dequantized[i] = (result.quantized[i] & 0xFF) * result.scale + result.minVal;
        }
        
        return dequantized;
    }
    
    /**
     * Calculate compression ratio
     * @param originalBits Original bits per value
     * @param quantizedBits Quantized bits per value
     * @return Compression ratio
     */
    public static double compressionRatio(int originalBits, int quantizedBits) {
        return (double) originalBits / quantizedBits;
    }
}

/**
 * Main demonstration class
 */
class VectorProcessingDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("Vector Processing Operations - Production Ready Examples");
        System.out.println("=".repeat(60));
        
        // Example 1: Cosine Similarity
        System.out.println("\n1. Cosine Similarity");
        double[] v1 = {1.0, 2.0, 3.0};
        double[] v2 = {4.0, 5.0, 6.0};
        double similarity = VectorOperations.cosineSimilarity(v1, v2);
        System.out.printf("   Vector 1: %s%n", Arrays.toString(v1));
        System.out.printf("   Vector 2: %s%n", Arrays.toString(v2));
        System.out.printf("   Cosine Similarity: %.4f%n", similarity);
        
        // Example 2: Distance Metrics
        System.out.println("\n2. Distance Metrics");
        double euclidean = VectorOperations.euclideanDistance(v1, v2);
        double manhattan = VectorOperations.manhattanDistance(v1, v2);
        System.out.printf("   Euclidean Distance: %.4f%n", euclidean);
        System.out.printf("   Manhattan Distance: %.4f%n", manhattan);
        
        // Example 3: Vector Database
        System.out.println("\n3. Vector Database (Similarity Search)");
        VectorDatabase db = new VectorDatabase(128);
        Random random = new Random(42);
        
        // Add sample vectors
        for (int i = 0; i < 100; i++) {
            double[] vec = new double[128];
            for (int j = 0; j < 128; j++) {
                vec[j] = random.nextGaussian();
            }
            vec = VectorOperations.normalizeL2(vec);
            
            Map<String, Object> meta = new HashMap<>();
            meta.put("id", i);
            meta.put("category", "cat_" + (i % 5));
            db.add(vec, meta);
        }
        
        // Search
        double[] query = new double[128];
        for (int i = 0; i < 128; i++) {
            query[i] = random.nextGaussian();
        }
        query = VectorOperations.normalizeL2(query);
        
        List<SearchResult> results = db.search(query, 5);
        System.out.printf("   Database size: %d vectors%n", db.size());
        System.out.println("   Top 5 similar vectors:");
        for (SearchResult result : results) {
            System.out.printf("      Index %d: Score %.4f, Category: %s%n", 
                            result.index, result.score, result.metadata.get("category"));
        }
        
        // Example 4: Vector Operations
        System.out.println("\n4. Vector Operations");
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 5.0, 6.0};
        double[] sum = VectorOperations.add(a, b);
        double[] diff = VectorOperations.subtract(a, b);
        double[] scaled = VectorOperations.scale(a, 2.0);
        System.out.printf("   a + b = %s%n", Arrays.toString(sum));
        System.out.printf("   a - b = %s%n", Arrays.toString(diff));
        System.out.printf("   2 * a = %s%n", Arrays.toString(scaled));
        
        // Example 5: Quantization
        System.out.println("\n5. Vector Quantization (Compression)");
        double[] originalVec = new double[512];
        for (int i = 0; i < originalVec.length; i++) {
            originalVec[i] = random.nextGaussian() * 10;
        }
        
        VectorQuantization.QuantizationResult qResult = 
            VectorQuantization.scalarQuantization(originalVec, 8);
        double[] dequantized = VectorQuantization.dequantize(qResult);
        
        double mse = 0.0;
        for (int i = 0; i < originalVec.length; i++) {
            double diff2 = originalVec[i] - dequantized[i];
            mse += diff2 * diff2;
        }
        mse /= originalVec.length;
        
        System.out.printf("   Original size: %d bytes (float64)%n", originalVec.length * 8);
        System.out.printf("   Quantized size: %d bytes (uint8)%n", qResult.quantized.length);
        System.out.printf("   Compression ratio: %.2fx%n", 
                         VectorQuantization.compressionRatio(64, 8));
        System.out.printf("   Reconstruction error (MSE): %.6f%n", mse);
        
        // Example 6: Embeddings
        System.out.println("\n6. Word Embeddings");
        String[] words = {"hello", "world", "machine", "learning"};
        List<double[]> embeddings = Arrays.stream(words)
            .map(word -> VectorEmbeddings.createWordEmbedding(word, 300))
            .collect(Collectors.toList());
        
        double[] sentenceEmb = VectorEmbeddings.averagePooling(embeddings);
        System.out.printf("   Word embeddings dimension: %d%n", embeddings.get(0).length);
        System.out.print("   Sentence embedding (avg pool): [");
        for (int i = 0; i < Math.min(5, sentenceEmb.length); i++) {
            System.out.printf("%.4f%s", sentenceEmb[i], i < 4 ? ", " : "");
        }
        System.out.println("...]");
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("All examples completed successfully!");
        System.out.println("=".repeat(60));
    }
}
