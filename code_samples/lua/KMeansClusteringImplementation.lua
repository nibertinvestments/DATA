--[[
K-Means Clustering Implementation in Lua
========================================

This module demonstrates production-ready K-Means clustering implementation in Lua
with comprehensive error handling, modern Lua patterns, and extensive documentation
for AI training datasets.

Key Features:
- Object-oriented design with Lua metatables and metamethods
- K-Means++ initialization for better cluster selection
- Multiple distance metrics (Euclidean, Manhattan, Cosine)
- Comprehensive convergence criteria and validation
- OpenResty/LuaJIT compatibility for high performance
- Memory-efficient data structures and algorithms
- Extensive documentation for AI learning
- Production-ready patterns with error handling

Author: AI Training Dataset
License: MIT
]]

local KMeansML = {}
KMeansML.__index = KMeansML

-- Dependencies (optional - fallback to pure Lua if not available)
local json = pcall(require, "cjson") and require("cjson") or nil
local ffi = pcall(require, "ffi") and require("ffi") or nil

-- Configuration constants
local DEFAULT_MAX_ITERATIONS = 300
local DEFAULT_TOLERANCE = 1e-4
local DEFAULT_N_INIT = 10
local RANDOM_SEED = 42

-- Utility functions
local function set_random_seed(seed)
    math.randomseed(seed or RANDOM_SEED)
    -- Warm up the random number generator
    for _ = 1, 10 do
        math.random()
    end
end

local function deep_copy(t)
    if type(t) ~= "table" then
        return t
    end
    local copy = {}
    for k, v in pairs(t) do
        copy[k] = deep_copy(v)
    end
    return copy
end

local function validate_number(value, name)
    if type(value) ~= "number" or value ~= value then -- NaN check
        error(string.format("Invalid %s: must be a finite number, got %s", name, tostring(value)))
    end
    if value == math.huge or value == -math.huge then
        error(string.format("Invalid %s: cannot be infinite", name))
    end
end

local function validate_positive_integer(value, name)
    validate_number(value, name)
    if value <= 0 or math.floor(value) ~= value then
        error(string.format("Invalid %s: must be a positive integer, got %s", name, tostring(value)))
    end
end

-- Distance metrics module
local Distance = {}

function Distance.euclidean(point1, point2)
    if #point1 ~= #point2 then
        error("Points must have the same dimensionality")
    end
    
    local sum = 0
    for i = 1, #point1 do
        local diff = point1[i] - point2[i]
        sum = sum + diff * diff
    end
    return math.sqrt(sum)
end

function Distance.manhattan(point1, point2)
    if #point1 ~= #point2 then
        error("Points must have the same dimensionality")
    end
    
    local sum = 0
    for i = 1, #point1 do
        sum = sum + math.abs(point1[i] - point2[i])
    end
    return sum
end

function Distance.cosine(point1, point2)
    if #point1 ~= #point2 then
        error("Points must have the same dimensionality")
    end
    
    local dot_product = 0
    local norm1 = 0
    local norm2 = 0
    
    for i = 1, #point1 do
        dot_product = dot_product + point1[i] * point2[i]
        norm1 = norm1 + point1[i] * point1[i]
        norm2 = norm2 + point2[i] * point2[i]
    end
    
    norm1 = math.sqrt(norm1)
    norm2 = math.sqrt(norm2)
    
    if norm1 == 0 or norm2 == 0 then
        return 0
    end
    
    local similarity = dot_product / (norm1 * norm2)
    return 1 - similarity  -- Convert similarity to distance
end

Distance.metrics = {
    euclidean = Distance.euclidean,
    manhattan = Distance.manhattan,
    cosine = Distance.cosine
}

-- Validation module
local Validation = {}

function Validation.validate_dataset(data)
    if type(data) ~= "table" or #data == 0 then
        error("Dataset must be a non-empty table")
    end
    
    local first_point = data[1]
    if type(first_point) ~= "table" or #first_point == 0 then
        error("Each data point must be a non-empty table of numbers")
    end
    
    local expected_dimensions = #first_point
    
    for i, point in ipairs(data) do
        if type(point) ~= "table" then
            error(string.format("Data point %d must be a table", i))
        end
        
        if #point ~= expected_dimensions then
            error(string.format("Data point %d has %d dimensions, expected %d", 
                               i, #point, expected_dimensions))
        end
        
        for j, value in ipairs(point) do
            validate_number(value, string.format("data point %d, dimension %d", i, j))
        end
    end
    
    return expected_dimensions
end

function Validation.validate_k(k, data_size)
    validate_positive_integer(k, "k (number of clusters)")
    
    if k > data_size then
        error(string.format("k (%d) cannot be greater than the number of data points (%d)", k, data_size))
    end
end

function Validation.validate_distance_metric(metric_name)
    if type(metric_name) ~= "string" then
        error("Distance metric must be a string")
    end
    
    if not Distance.metrics[metric_name] then
        local available = {}
        for name in pairs(Distance.metrics) do
            table.insert(available, name)
        end
        error(string.format("Unknown distance metric '%s'. Available metrics: %s", 
                           metric_name, table.concat(available, ", ")))
    end
end

-- Statistics module
local Statistics = {}

function Statistics.mean(values)
    if #values == 0 then
        return 0
    end
    
    local sum = 0
    for _, value in ipairs(values) do
        sum = sum + value
    end
    return sum / #values
end

function Statistics.std_dev(values)
    if #values <= 1 then
        return 0
    end
    
    local mean_val = Statistics.mean(values)
    local sum_squares = 0
    
    for _, value in ipairs(values) do
        local diff = value - mean_val
        sum_squares = sum_squares + diff * diff
    end
    
    return math.sqrt(sum_squares / (#values - 1))
end

function Statistics.centroid(points)
    if #points == 0 then
        error("Cannot compute centroid of empty point set")
    end
    
    local dimensions = #points[1]
    local centroid = {}
    
    -- Initialize centroid
    for d = 1, dimensions do
        centroid[d] = 0
    end
    
    -- Sum all points
    for _, point in ipairs(points) do
        for d = 1, dimensions do
            centroid[d] = centroid[d] + point[d]
        end
    end
    
    -- Average
    for d = 1, dimensions do
        centroid[d] = centroid[d] / #points
    end
    
    return centroid
end

-- Main KMeans class
function KMeansML.new(config)
    config = config or {}
    
    local self = {
        k = config.k or 3,
        max_iterations = config.max_iterations or DEFAULT_MAX_ITERATIONS,
        tolerance = config.tolerance or DEFAULT_TOLERANCE,
        distance_metric = config.distance_metric or "euclidean",
        n_init = config.n_init or DEFAULT_N_INIT,
        random_seed = config.random_seed or RANDOM_SEED,
        verbose = config.verbose or false,
        
        -- Training state
        centroids = {},
        labels = {},
        inertia = nil,
        n_iterations = nil,
        converged = false,
        training_metrics = {},
        
        -- Data properties
        data = {},
        n_samples = 0,
        n_features = 0
    }
    
    setmetatable(self, KMeansML)
    
    -- Validate configuration
    self:_validate_config()
    
    return self
end

function KMeansML:_validate_config()
    validate_positive_integer(self.k, "k")
    validate_positive_integer(self.max_iterations, "max_iterations")
    validate_number(self.tolerance, "tolerance")
    validate_positive_integer(self.n_init, "n_init")
    Validation.validate_distance_metric(self.distance_metric)
    
    if self.tolerance <= 0 then
        error("tolerance must be positive")
    end
end

function KMeansML:_log(message)
    if self.verbose then
        print(string.format("[KMeans] %s", message))
    end
end

function KMeansML:fit(data)
    local start_time = os.clock()
    
    self:_log("Starting K-Means clustering...")
    
    -- Validate and store data
    self.n_features = Validation.validate_dataset(data)
    self.n_samples = #data
    Validation.validate_k(self.k, self.n_samples)
    self.data = deep_copy(data)
    
    self:_log(string.format("Dataset: %d samples, %d features, k=%d", 
                           self.n_samples, self.n_features, self.k))
    self:_log(string.format("Distance metric: %s", self.distance_metric))
    
    set_random_seed(self.random_seed)
    
    -- Multiple initialization runs
    local best_inertia = math.huge
    local best_centroids = {}
    local best_labels = {}
    local best_iterations = 0
    
    for init_run = 1, self.n_init do
        self:_log(string.format("Initialization run %d/%d", init_run, self.n_init))
        
        local centroids = self:_initialize_centroids_plus_plus()
        local labels, inertia, iterations, converged = self:_fit_single_run(centroids)
        
        if inertia < best_inertia then
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels
            best_iterations = iterations
            self.converged = converged
        end
        
        self:_log(string.format("Run %d: inertia=%.6f, iterations=%d, converged=%s", 
                               init_run, inertia, iterations, tostring(converged)))
    end
    
    -- Store best results
    self.centroids = best_centroids
    self.labels = best_labels
    self.inertia = best_inertia
    self.n_iterations = best_iterations
    
    local training_time = os.clock() - start_time
    
    -- Calculate training metrics
    self.training_metrics = self:_calculate_training_metrics(training_time)
    
    self:_log(string.format("Training completed in %.3f seconds", training_time))
    self:_log(string.format("Final inertia: %.6f", self.inertia))
    self:_log(string.format("Converged: %s", tostring(self.converged)))
    
    return self
end

function KMeansML:_initialize_centroids_plus_plus()
    local centroids = {}
    local distance_func = Distance.metrics[self.distance_metric]
    
    -- Choose first centroid randomly
    local first_idx = math.random(1, self.n_samples)
    centroids[1] = deep_copy(self.data[first_idx])
    
    self:_log("K-Means++ initialization...")
    
    -- Choose remaining centroids
    for c = 2, self.k do
        local distances = {}
        local total_distance = 0
        
        -- Calculate minimum distance to existing centroids for each point
        for i, point in ipairs(self.data) do
            local min_distance = math.huge
            
            for _, centroid in ipairs(centroids) do
                local dist = distance_func(point, centroid)
                min_distance = math.min(min_distance, dist)
            end
            
            distances[i] = min_distance * min_distance  -- Square for probability weighting
            total_distance = total_distance + distances[i]
        end
        
        -- Choose next centroid with probability proportional to squared distance
        local target = math.random() * total_distance
        local cumulative = 0
        
        for i, dist_sq in ipairs(distances) do
            cumulative = cumulative + dist_sq
            if cumulative >= target then
                centroids[c] = deep_copy(self.data[i])
                break
            end
        end
    end
    
    return centroids
end

function KMeansML:_fit_single_run(initial_centroids)
    local centroids = deep_copy(initial_centroids)
    local labels = {}
    local distance_func = Distance.metrics[self.distance_metric]
    
    for iteration = 1, self.max_iterations do
        local changed = false
        
        -- Assignment step: assign points to nearest centroids
        for i, point in ipairs(self.data) do
            local min_distance = math.huge
            local best_cluster = 1
            
            for c, centroid in ipairs(centroids) do
                local dist = distance_func(point, centroid)
                if dist < min_distance then
                    min_distance = dist
                    best_cluster = c
                end
            end
            
            if labels[i] ~= best_cluster then
                changed = true
                labels[i] = best_cluster
            end
        end
        
        -- Update step: recalculate centroids
        local new_centroids = {}
        local cluster_sizes = {}
        
        -- Initialize cluster statistics
        for c = 1, self.k do
            new_centroids[c] = {}
            for d = 1, self.n_features do
                new_centroids[c][d] = 0
            end
            cluster_sizes[c] = 0
        end
        
        -- Accumulate points by cluster
        for i, point in ipairs(self.data) do
            local cluster = labels[i]
            cluster_sizes[cluster] = cluster_sizes[cluster] + 1
            
            for d = 1, self.n_features do
                new_centroids[cluster][d] = new_centroids[cluster][d] + point[d]
            end
        end
        
        -- Calculate new centroids (means)
        local centroid_shift = 0
        for c = 1, self.k do
            if cluster_sizes[c] > 0 then
                for d = 1, self.n_features do
                    new_centroids[c][d] = new_centroids[c][d] / cluster_sizes[c]
                end
                
                -- Calculate centroid shift
                local shift = distance_func(centroids[c], new_centroids[c])
                centroid_shift = math.max(centroid_shift, shift)
            else
                -- Handle empty cluster: reinitialize randomly
                local random_idx = math.random(1, self.n_samples)
                new_centroids[c] = deep_copy(self.data[random_idx])
                centroid_shift = math.huge -- Force continuation
            end
        end
        
        centroids = new_centroids
        
        -- Check convergence
        if centroid_shift < self.tolerance then
            local inertia = self:_calculate_inertia(centroids, labels)
            return labels, inertia, iteration, true
        end
        
        if iteration % 50 == 0 then
            self:_log(string.format("Iteration %d: centroid shift=%.6f", iteration, centroid_shift))
        end
    end
    
    -- Max iterations reached
    local inertia = self:_calculate_inertia(centroids, labels)
    return labels, inertia, self.max_iterations, false
end

function KMeansML:_calculate_inertia(centroids, labels)
    local inertia = 0
    local distance_func = Distance.metrics[self.distance_metric]
    
    for i, point in ipairs(self.data) do
        local cluster = labels[i]
        local centroid = centroids[cluster]
        local dist = distance_func(point, centroid)
        inertia = inertia + dist * dist
    end
    
    return inertia
end

function KMeansML:predict(data)
    if not self.centroids or #self.centroids == 0 then
        error("Model must be fitted before prediction")
    end
    
    Validation.validate_dataset(data)
    
    if #data[1] ~= self.n_features then
        error(string.format("Input data has %d features, expected %d", #data[1], self.n_features))
    end
    
    local predictions = {}
    local distance_func = Distance.metrics[self.distance_metric]
    
    for i, point in ipairs(data) do
        local min_distance = math.huge
        local best_cluster = 1
        
        for c, centroid in ipairs(self.centroids) do
            local dist = distance_func(point, centroid)
            if dist < min_distance then
                min_distance = dist
                best_cluster = c
            end
        end
        
        predictions[i] = best_cluster
    end
    
    return predictions
end

function KMeansML:predict_single(point)
    local predictions = self:predict({point})
    return predictions[1]
end

function KMeansML:transform(data)
    if not self.centroids or #self.centroids == 0 then
        error("Model must be fitted before transformation")
    end
    
    Validation.validate_dataset(data)
    
    local distances = {}
    local distance_func = Distance.metrics[self.distance_metric]
    
    for i, point in ipairs(data) do
        distances[i] = {}
        for c, centroid in ipairs(self.centroids) do
            distances[i][c] = distance_func(point, centroid)
        end
    end
    
    return distances
end

function KMeansML:fit_predict(data)
    self:fit(data)
    return self.labels
end

function KMeansML:_calculate_training_metrics(training_time)
    local metrics = {
        inertia = self.inertia,
        n_iterations = self.n_iterations,
        converged = self.converged,
        training_time = training_time,
        n_samples = self.n_samples,
        n_features = self.n_features,
        k = self.k
    }
    
    -- Calculate cluster statistics
    local cluster_sizes = {}
    local cluster_inertias = {}
    
    for c = 1, self.k do
        cluster_sizes[c] = 0
        cluster_inertias[c] = 0
    end
    
    local distance_func = Distance.metrics[self.distance_metric]
    
    for i, point in ipairs(self.data) do
        local cluster = self.labels[i]
        cluster_sizes[cluster] = cluster_sizes[cluster] + 1
        
        local centroid = self.centroids[cluster]
        local dist = distance_func(point, centroid)
        cluster_inertias[cluster] = cluster_inertias[cluster] + dist * dist
    end
    
    metrics.cluster_sizes = cluster_sizes
    metrics.cluster_inertias = cluster_inertias
    
    -- Calculate silhouette coefficient (simplified)
    if self.n_samples <= 1000 then  -- Only for small datasets due to O(n²) complexity
        metrics.silhouette_score = self:_calculate_silhouette_score()
    end
    
    return metrics
end

function KMeansML:_calculate_silhouette_score()
    if self.k == 1 then
        return 0  -- Silhouette score is undefined for k=1
    end
    
    local silhouette_scores = {}
    local distance_func = Distance.metrics[self.distance_metric]
    
    for i, point in ipairs(self.data) do
        local cluster = self.labels[i]
        
        -- Calculate average intra-cluster distance (a)
        local intra_distances = {}
        for j, other_point in ipairs(self.data) do
            if i ~= j and self.labels[j] == cluster then
                table.insert(intra_distances, distance_func(point, other_point))
            end
        end
        
        local a = #intra_distances > 0 and Statistics.mean(intra_distances) or 0
        
        -- Calculate minimum average inter-cluster distance (b)
        local b = math.huge
        
        for c = 1, self.k do
            if c ~= cluster then
                local inter_distances = {}
                for j, other_point in ipairs(self.data) do
                    if self.labels[j] == c then
                        table.insert(inter_distances, distance_func(point, other_point))
                    end
                end
                
                if #inter_distances > 0 then
                    local avg_inter_dist = Statistics.mean(inter_distances)
                    b = math.min(b, avg_inter_dist)
                end
            end
        end
        
        -- Calculate silhouette coefficient for this point
        local silhouette
        if a == 0 and b == math.huge then
            silhouette = 0
        else
            silhouette = (b - a) / math.max(a, b)
        end
        
        silhouette_scores[i] = silhouette
    end
    
    return Statistics.mean(silhouette_scores)
end

function KMeansML:get_cluster_centers()
    return deep_copy(self.centroids)
end

function KMeansML:get_labels()
    return deep_copy(self.labels)
end

function KMeansML:get_inertia()
    return self.inertia
end

function KMeansML:get_metrics()
    return deep_copy(self.training_metrics)
end

-- Model serialization
function KMeansML:to_json()
    if not json then
        error("JSON library not available - cannot serialize model")
    end
    
    local model_data = {
        centroids = self.centroids,
        k = self.k,
        distance_metric = self.distance_metric,
        n_features = self.n_features,
        training_metrics = self.training_metrics,
        config = {
            max_iterations = self.max_iterations,
            tolerance = self.tolerance,
            n_init = self.n_init,
            random_seed = self.random_seed
        }
    }
    
    return json.encode(model_data)
end

function KMeansML:from_json(json_string)
    if not json then
        error("JSON library not available - cannot deserialize model")
    end
    
    local model_data = json.decode(json_string)
    
    self.centroids = model_data.centroids
    self.k = model_data.k
    self.distance_metric = model_data.distance_metric
    self.n_features = model_data.n_features
    self.training_metrics = model_data.training_metrics
    
    if model_data.config then
        local config = model_data.config
        self.max_iterations = config.max_iterations
        self.tolerance = config.tolerance
        self.n_init = config.n_init
        self.random_seed = config.random_seed
    end
    
    return self
end

-- Data generation utilities
local DataUtils = {}

function DataUtils.generate_blobs(n_samples, n_centers, n_features, center_box, cluster_std, random_seed)
    n_samples = n_samples or 100
    n_centers = n_centers or 3
    n_features = n_features or 2
    center_box = center_box or {-10, 10}
    cluster_std = cluster_std or 1.0
    random_seed = random_seed or RANDOM_SEED
    
    set_random_seed(random_seed)
    
    -- Generate random centers
    local centers = {}
    for c = 1, n_centers do
        centers[c] = {}
        for d = 1, n_features do
            local range = center_box[2] - center_box[1]
            centers[c][d] = center_box[1] + math.random() * range
        end
    end
    
    -- Generate samples around centers
    local data = {}
    local true_labels = {}
    local samples_per_cluster = math.floor(n_samples / n_centers)
    local sample_idx = 1
    
    for c = 1, n_centers do
        local cluster_samples = (c == n_centers) and (n_samples - sample_idx + 1) or samples_per_cluster
        
        for _ = 1, cluster_samples do
            local point = {}
            for d = 1, n_features do
                -- Generate point with Gaussian noise around center
                local u1 = math.random()
                local u2 = math.random()
                local normal = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)  -- Box-Muller transform
                point[d] = centers[c][d] + normal * cluster_std
            end
            
            data[sample_idx] = point
            true_labels[sample_idx] = c
            sample_idx = sample_idx + 1
        end
    end
    
    return data, true_labels, centers
end

function DataUtils.generate_circles(n_samples, noise, factor, random_seed)
    n_samples = n_samples or 100
    noise = noise or 0.05
    factor = factor or 0.8
    random_seed = random_seed or RANDOM_SEED
    
    set_random_seed(random_seed)
    
    local data = {}
    local labels = {}
    local samples_per_circle = math.floor(n_samples / 2)
    
    -- Outer circle
    for i = 1, samples_per_circle do
        local angle = 2 * math.pi * math.random()
        local radius = 1.0 + (math.random() - 0.5) * noise
        
        data[i] = {
            radius * math.cos(angle),
            radius * math.sin(angle)
        }
        labels[i] = 1
    end
    
    -- Inner circle
    for i = samples_per_circle + 1, n_samples do
        local angle = 2 * math.pi * math.random()
        local radius = factor + (math.random() - 0.5) * noise
        
        data[i] = {
            radius * math.cos(angle),
            radius * math.sin(angle)
        }
        labels[i] = 2
    end
    
    return data, labels
end

function DataUtils.make_random_dataset(n_samples, n_features, n_clusters, random_seed)
    n_samples = n_samples or 300
    n_features = n_features or 2
    n_clusters = n_clusters or 3
    random_seed = random_seed or RANDOM_SEED
    
    return DataUtils.generate_blobs(n_samples, n_clusters, n_features, {-5, 5}, 1.5, random_seed)
end

-- Demo module
local Demo = {}

function Demo.run_kmeans_demo()
    print("🎯 Lua K-Means Clustering Implementation Demo")
    print(string.rep("=", 50))
    
    local function demo_step(name, func)
        print(string.format("\n🔹 %s", name))
        local success, result = pcall(func)
        if not success then
            print(string.format("❌ %s failed: %s", name, result))
            return false
        end
        return true
    end
    
    -- Demo 1: Basic clustering with synthetic blob data
    demo_step("Generating synthetic blob dataset", function()
        local data, true_labels = DataUtils.generate_blobs(300, 4, 2, {-10, 10}, 2.0)
        print(string.format("📊 Generated %d samples with 4 true clusters", #data))
        
        local kmeans = KMeansML.new({
            k = 4,
            max_iterations = 300,
            tolerance = 1e-4,
            distance_metric = "euclidean",
            n_init = 5,
            verbose = true
        })
        
        kmeans:fit(data)
        
        local metrics = kmeans:get_metrics()
        print(string.format("✅ Training completed: inertia=%.4f, iterations=%d", 
                           metrics.inertia, metrics.n_iterations))
        print(string.format("✅ Converged: %s", tostring(metrics.converged)))
        
        if metrics.silhouette_score then
            print(string.format("✅ Silhouette score: %.4f", metrics.silhouette_score))
        end
        
        -- Test prediction
        local test_point = {0, 0}
        local predicted_cluster = kmeans:predict_single(test_point)
        print(string.format("🔮 Point [0, 0] assigned to cluster %d", predicted_cluster))
        
        -- Test multiple distance metrics
        print("\n📏 Testing different distance metrics:")
        local metrics_comparison = {}
        
        for metric_name in pairs(Distance.metrics) do
            local test_kmeans = KMeansML.new({
                k = 4,
                distance_metric = metric_name,
                verbose = false,
                n_init = 3
            })
            
            test_kmeans:fit(data)
            local test_metrics = test_kmeans:get_metrics()
            metrics_comparison[metric_name] = test_metrics.inertia
            
            print(string.format("   %s: inertia=%.4f", metric_name, test_metrics.inertia))
        end
    end)
    
    -- Demo 2: Circular dataset clustering
    demo_step("Clustering circular dataset", function()
        local data, true_labels = DataUtils.generate_circles(200, 0.1, 0.6)
        print(string.format("📊 Generated circular dataset with %d samples", #data))
        
        local kmeans = KMeansML.new({
            k = 2,
            distance_metric = "euclidean",
            verbose = true,
            n_init = 10
        })
        
        kmeans:fit(data)
        
        local metrics = kmeans:get_metrics()
        print(string.format("✅ Circular clustering: inertia=%.4f", metrics.inertia))
        
        -- Show cluster centers
        local centers = kmeans:get_cluster_centers()
        print("📍 Cluster centers:")
        for i, center in ipairs(centers) do
            print(string.format("   Cluster %d: [%.3f, %.3f]", 
                               i, center[1], center[2]))
        end
    end)
    
    -- Demo 3: Model serialization
    demo_step("Model serialization test", function()
        if not json then
            print("⚠️ JSON library not available - skipping serialization test")
            return
        end
        
        local data = DataUtils.make_random_dataset(150, 3, 3)
        local kmeans = KMeansML.new({k = 3, verbose = false})
        kmeans:fit(data)
        
        -- Serialize model
        local json_model = kmeans:to_json()
        print(string.format("💾 Model serialized to %d characters", #json_model))
        
        -- Create new model and deserialize
        local new_kmeans = KMeansML.new({})
        new_kmeans:from_json(json_model)
        
        -- Test that predictions are identical
        local test_point = data[1]
        local original_pred = kmeans:predict_single(test_point)
        local loaded_pred = new_kmeans:predict_single(test_point)
        
        print(string.format("🔄 Original prediction: %d", original_pred))
        print(string.format("🔄 Loaded prediction: %d", loaded_pred))
        print(string.format("✅ Serialization test: %s", 
                           original_pred == loaded_pred and "PASSED" or "FAILED"))
    end)
    
    -- Demo 4: Performance test
    demo_step("Performance benchmark", function()
        print("⚡ Running performance benchmark...")
        
        local sizes = {100, 500, 1000}
        local results = {}
        
        for _, size in ipairs(sizes) do
            local data = DataUtils.make_random_dataset(size, 5, 4)
            
            local start_time = os.clock()
            local kmeans = KMeansML.new({
                k = 4,
                verbose = false,
                n_init = 5
            })
            kmeans:fit(data)
            local end_time = os.clock()
            
            local metrics = kmeans:get_metrics()
            results[size] = {
                time = end_time - start_time,
                inertia = metrics.inertia,
                iterations = metrics.n_iterations
            }
            
            print(string.format("   %d samples: %.3fs, %d iterations, inertia=%.2f", 
                               size, results[size].time, results[size].iterations, 
                               results[size].inertia))
        end
    end)
    
    print("\n✅ K-Means clustering demonstration completed successfully!")
    print("\n📋 Summary of implemented features:")
    print("   • K-Means++ initialization for better cluster selection")
    print("   • Multiple distance metrics (Euclidean, Manhattan, Cosine)")
    print("   • Multiple random initializations with best result selection")
    print("   • Comprehensive convergence criteria and validation")
    print("   • Silhouette score calculation for cluster quality assessment")
    print("   • Model serialization and deserialization with JSON")
    print("   • Synthetic dataset generation utilities")
    print("   • Production-ready error handling and logging")
    print("   • Memory-efficient implementation with LuaJIT compatibility")
end

-- Export modules
return {
    KMeansML = KMeansML,
    Distance = Distance,
    DataUtils = DataUtils,
    Demo = Demo
}

-- Run demo if this file is executed directly
if arg and arg[0] and arg[0]:match("KMeansClusteringImplementation%.lua$") then
    Demo.run_kmeans_demo()
end