/**
 * Advanced Portfolio Risk Management System
 * Java implementation of comprehensive risk calculation methods
 * 
 * Features:
 * - Value at Risk (VaR) calculation using multiple methods
 * - Conditional Value at Risk (CVaR) 
 * - Maximum Drawdown analysis
 * - Risk decomposition and attribution
 * - Stress testing capabilities
 */

import java.util.*;
import java.util.stream.*;
import java.lang.Math;

public class PortfolioRiskManager {
    
    public static class RiskMetrics {
        public final double var95;
        public final double var99;
        public final double cvar95;
        public final double cvar99;
        public final double maxDrawdown;
        public final double volatility;
        public final double sharpeRatio;
        public final double sortinoRatio;
        public final double calmarRatio;
        public final Double beta;
        public final Double trackingError;
        
        public RiskMetrics(double var95, double var99, double cvar95, double cvar99,
                          double maxDrawdown, double volatility, double sharpeRatio,
                          double sortinoRatio, double calmarRatio, Double beta, Double trackingError) {
            this.var95 = var95;
            this.var99 = var99;
            this.cvar95 = cvar95;
            this.cvar99 = cvar99;
            this.maxDrawdown = maxDrawdown;
            this.volatility = volatility;
            this.sharpeRatio = sharpeRatio;
            this.sortinoRatio = sortinoRatio;
            this.calmarRatio = calmarRatio;
            this.beta = beta;
            this.trackingError = trackingError;
        }
        
        @Override
        public String toString() {
            return String.format(
                "RiskMetrics{var95=%.6f, var99=%.6f, cvar95=%.6f, cvar99=%.6f, " +
                "maxDrawdown=%.6f, volatility=%.6f, sharpeRatio=%.6f, sortinoRatio=%.6f, " +
                "calmarRatio=%.6f, beta=%.6f, trackingError=%.6f}",
                var95, var99, cvar95, cvar99, maxDrawdown, volatility, 
                sharpeRatio, sortinoRatio, calmarRatio, 
                beta != null ? beta : Double.NaN, 
                trackingError != null ? trackingError : Double.NaN
            );
        }
    }
    
    public static class DrawdownMetrics {
        public final double maxDrawdown;
        public final int drawdownDuration;
        public final Integer recoveryTime;
        public final double currentDrawdown;
        public final double averageDrawdown;
        
        public DrawdownMetrics(double maxDrawdown, int drawdownDuration, 
                              Integer recoveryTime, double currentDrawdown, double averageDrawdown) {
            this.maxDrawdown = maxDrawdown;
            this.drawdownDuration = drawdownDuration;
            this.recoveryTime = recoveryTime;
            this.currentDrawdown = currentDrawdown;
            this.averageDrawdown = averageDrawdown;
        }
    }
    
    private final double[] returns;
    private final double[] benchmarkReturns;
    private final double meanReturn;
    private final double volatility;
    
    public PortfolioRiskManager(double[] returns, double[] benchmarkReturns) {
        this.returns = Arrays.copyOf(returns, returns.length);
        this.benchmarkReturns = benchmarkReturns != null ? 
            Arrays.copyOf(benchmarkReturns, benchmarkReturns.length) : null;
        
        // Calculate basic statistics
        this.meanReturn = Arrays.stream(returns).average().orElse(0.0);
        
        double sumSquaredDeviations = Arrays.stream(returns)
            .map(r -> Math.pow(r - meanReturn, 2))
            .sum();
        this.volatility = Math.sqrt(sumSquaredDeviations / (returns.length - 1));
    }
    
    public PortfolioRiskManager(double[] returns) {
        this(returns, null);
    }
    
    /**
     * Calculate Value at Risk using historical method
     */
    public double calculateHistoricalVaR(double confidenceLevel) {
        double[] sortedReturns = Arrays.copyOf(returns, returns.length);
        Arrays.sort(sortedReturns);
        
        double alpha = 1.0 - confidenceLevel;
        int index = (int) Math.ceil(alpha * sortedReturns.length) - 1;
        index = Math.max(0, Math.min(index, sortedReturns.length - 1));
        
        return -sortedReturns[index];
    }
    
    /**
     * Calculate parametric VaR assuming normal distribution
     */
    public double calculateParametricVaR(double confidenceLevel) {
        double alpha = 1.0 - confidenceLevel;
        double zScore = inverseNormalCDF(alpha);
        return -(meanReturn + zScore * volatility);
    }
    
    /**
     * Calculate Monte Carlo VaR
     */
    public double calculateMonteCarloVaR(double confidenceLevel, int numSimulations) {
        Random random = new Random(42); // Fixed seed for reproducibility
        double[] simulatedReturns = new double[numSimulations];
        
        for (int i = 0; i < numSimulations; i++) {
            simulatedReturns[i] = meanReturn + volatility * random.nextGaussian();
        }
        
        Arrays.sort(simulatedReturns);
        double alpha = 1.0 - confidenceLevel;
        int index = (int) Math.ceil(alpha * numSimulations) - 1;
        index = Math.max(0, Math.min(index, numSimulations - 1));
        
        return -simulatedReturns[index];
    }
    
    /**
     * Calculate Conditional Value at Risk (Expected Shortfall)
     */
    public double calculateCVaR(double confidenceLevel, String method) {
        double var = 0.0;
        
        switch (method.toLowerCase()) {
            case "historical":
                var = calculateHistoricalVaR(confidenceLevel);
                double[] lossesArray = Arrays.stream(returns)
                    .filter(r -> -r >= var)
                    .map(r -> -r)
                    .toArray();
                return lossesArray.length > 0 ? 
                    Arrays.stream(lossesArray).average().orElse(var) : var;
                
            case "parametric":
                double alpha = 1.0 - confidenceLevel;
                double zAlpha = inverseNormalCDF(alpha);
                double phiZ = normalPDF(zAlpha);
                return -(meanReturn - volatility * phiZ / alpha);
                
            case "monte_carlo":
                var = calculateMonteCarloVaR(confidenceLevel, 10000);
                Random random = new Random(42);
                double[] simReturns = new double[10000];
                for (int i = 0; i < 10000; i++) {
                    simReturns[i] = meanReturn + volatility * random.nextGaussian();
                }
                double[] losses = Arrays.stream(simReturns)
                    .filter(r -> -r >= var)
                    .map(r -> -r)
                    .toArray();
                return losses.length > 0 ? 
                    Arrays.stream(losses).average().orElse(var) : var;
                
            default:
                throw new IllegalArgumentException("Unknown method: " + method);
        }
    }
    
    /**
     * Calculate maximum drawdown and related metrics
     */
    public DrawdownMetrics calculateMaximumDrawdown() {
        double[] cumulativeReturns = new double[returns.length];
        cumulativeReturns[0] = 1.0 + returns[0];
        
        for (int i = 1; i < returns.length; i++) {
            cumulativeReturns[i] = cumulativeReturns[i-1] * (1.0 + returns[i]);
        }
        
        double[] runningMax = new double[cumulativeReturns.length];
        runningMax[0] = cumulativeReturns[0];
        
        for (int i = 1; i < cumulativeReturns.length; i++) {
            runningMax[i] = Math.max(runningMax[i-1], cumulativeReturns[i]);
        }
        
        double[] drawdowns = new double[cumulativeReturns.length];
        for (int i = 0; i < cumulativeReturns.length; i++) {
            drawdowns[i] = (cumulativeReturns[i] - runningMax[i]) / runningMax[i];
        }
        
        // Find maximum drawdown
        double maxDrawdown = Arrays.stream(drawdowns).min().orElse(0.0);
        int maxDdIndex = IntStream.range(0, drawdowns.length)
            .reduce((i, j) -> drawdowns[i] < drawdowns[j] ? i : j)
            .orElse(0);
        
        // Find peak before max drawdown
        int peakIndex = IntStream.range(0, maxDdIndex + 1)
            .reduce((i, j) -> runningMax[i] > runningMax[j] ? i : j)
            .orElse(0);
        
        // Calculate recovery time
        Integer recoveryIndex = null;
        for (int i = maxDdIndex; i < cumulativeReturns.length; i++) {
            if (cumulativeReturns[i] >= runningMax[maxDdIndex]) {
                recoveryIndex = i;
                break;
            }
        }
        
        Integer recoveryTime = recoveryIndex != null ? recoveryIndex - maxDdIndex : null;
        double currentDrawdown = Math.abs(drawdowns[drawdowns.length - 1]);
        double averageDrawdown = Arrays.stream(drawdowns)
            .map(Math::abs)
            .average().orElse(0.0);
        
        return new DrawdownMetrics(
            Math.abs(maxDrawdown),
            maxDdIndex - peakIndex,
            recoveryTime,
            currentDrawdown,
            averageDrawdown
        );
    }
    
    /**
     * Calculate comprehensive risk metrics
     */
    public RiskMetrics calculateRiskMetrics(double riskFreeRate) {
        // VaR calculations
        double var95 = calculateHistoricalVaR(0.95);
        double var99 = calculateHistoricalVaR(0.99);
        double cvar95 = calculateCVaR(0.95, "historical");
        double cvar99 = calculateCVaR(0.99, "historical");
        
        // Drawdown
        DrawdownMetrics ddMetrics = calculateMaximumDrawdown();
        double maxDrawdown = ddMetrics.maxDrawdown;
        
        // Performance ratios
        double excessReturn = meanReturn - riskFreeRate;
        double sharpeRatio = volatility > 0 ? excessReturn / volatility : 0.0;
        
        // Sortino ratio (downside deviation)
        double[] downsideReturns = Arrays.stream(returns)
            .filter(r -> r < 0)
            .toArray();
        
        double downsideDeviation = 0.0;
        if (downsideReturns.length > 0) {
            double downsideMean = Arrays.stream(downsideReturns).average().orElse(0.0);
            double sumSquaredDeviations = Arrays.stream(downsideReturns)
                .map(r -> Math.pow(r - downsideMean, 2))
                .sum();
            downsideDeviation = Math.sqrt(sumSquaredDeviations / downsideReturns.length);
        }
        
        double sortinoRatio = downsideDeviation > 0 ? excessReturn / downsideDeviation : 0.0;
        
        // Calmar ratio
        double calmarRatio = maxDrawdown > 0 ? meanReturn / maxDrawdown : 0.0;
        
        // Beta and tracking error (if benchmark provided)
        Double beta = null;
        Double trackingError = null;
        
        if (benchmarkReturns != null && benchmarkReturns.length == returns.length) {
            beta = calculateBeta();
            trackingError = calculateTrackingError();
        }
        
        return new RiskMetrics(
            var95, var99, cvar95, cvar99, maxDrawdown, volatility,
            sharpeRatio, sortinoRatio, calmarRatio, beta, trackingError
        );
    }
    
    private double calculateBeta() {
        if (benchmarkReturns == null) return Double.NaN;
        
        double benchmarkMean = Arrays.stream(benchmarkReturns).average().orElse(0.0);
        
        double covariance = 0.0;
        double benchmarkVariance = 0.0;
        
        for (int i = 0; i < Math.min(returns.length, benchmarkReturns.length); i++) {
            double portfolioDeviation = returns[i] - meanReturn;
            double benchmarkDeviation = benchmarkReturns[i] - benchmarkMean;
            
            covariance += portfolioDeviation * benchmarkDeviation;
            benchmarkVariance += benchmarkDeviation * benchmarkDeviation;
        }
        
        int n = Math.min(returns.length, benchmarkReturns.length);
        covariance /= (n - 1);
        benchmarkVariance /= (n - 1);
        
        return benchmarkVariance > 0 ? covariance / benchmarkVariance : 0.0;
    }
    
    private double calculateTrackingError() {
        if (benchmarkReturns == null) return Double.NaN;
        
        double[] activeReturns = new double[Math.min(returns.length, benchmarkReturns.length)];
        for (int i = 0; i < activeReturns.length; i++) {
            activeReturns[i] = returns[i] - benchmarkReturns[i];
        }
        
        double activeMean = Arrays.stream(activeReturns).average().orElse(0.0);
        double sumSquaredDeviations = Arrays.stream(activeReturns)
            .map(r -> Math.pow(r - activeMean, 2))
            .sum();
        
        return Math.sqrt(sumSquaredDeviations / (activeReturns.length - 1));
    }
    
    /**
     * Stress test portfolio under different scenarios
     */
    public Map<String, Map<String, Double>> stressTest(Map<String, Map<String, Double>> scenarios) {
        Map<String, Map<String, Double>> results = new HashMap<>();
        
        for (Map.Entry<String, Map<String, Double>> scenario : scenarios.entrySet()) {
            String scenarioName = scenario.getKey();
            Map<String, Double> scenarioParams = scenario.getValue();
            
            double meanShock = scenarioParams.getOrDefault("mean_shock", 0.0);
            double volShock = scenarioParams.getOrDefault("vol_shock", 1.0);
            
            // Apply shocks to returns
            double[] stressedReturns = Arrays.stream(returns)
                .map(r -> (r + meanShock) * volShock)
                .toArray();
            
            // Create temporary risk manager for stressed scenario
            PortfolioRiskManager tempManager = new PortfolioRiskManager(stressedReturns);
            
            // Calculate metrics under stress
            double var95Stress = tempManager.calculateHistoricalVaR(0.95);
            double cvar95Stress = tempManager.calculateCVaR(0.95, "historical");
            DrawdownMetrics ddStress = tempManager.calculateMaximumDrawdown();
            
            Map<String, Double> scenarioResults = new HashMap<>();
            scenarioResults.put("var_95", var95Stress);
            scenarioResults.put("cvar_95", cvar95Stress);
            scenarioResults.put("max_drawdown", ddStress.maxDrawdown);
            scenarioResults.put("volatility", tempManager.volatility);
            scenarioResults.put("mean_return", tempManager.meanReturn);
            
            results.put(scenarioName, scenarioResults);
        }
        
        return results;
    }
    
    // Utility methods for statistical calculations
    private static double inverseNormalCDF(double p) {
        // Beasley-Springer-Moro algorithm approximation
        double a0 = 2.515517, a1 = 0.802853, a2 = 0.010328;
        double b1 = 1.432788, b2 = 0.189269, b3 = 0.001308;
        
        if (p >= 0.5) {
            double t = Math.sqrt(-2.0 * Math.log(1.0 - p));
            return t - (a0 + a1 * t + a2 * t * t) / (1.0 + b1 * t + b2 * t * t + b3 * t * t * t);
        } else {
            double t = Math.sqrt(-2.0 * Math.log(p));
            return -(t - (a0 + a1 * t + a2 * t * t) / (1.0 + b1 * t + b2 * t * t + b3 * t * t * t));
        }
    }
    
    private static double normalPDF(double x) {
        return Math.exp(-0.5 * x * x) / Math.sqrt(2.0 * Math.PI);
    }
    
    // Example usage and testing
    public static void main(String[] args) {
        System.out.println("=== Portfolio Risk Management Example (Java) ===");
        
        // Generate synthetic portfolio returns
        Random random = new Random(42);
        int nPeriods = 1000;
        
        // Generate correlated returns
        double[] portfolioReturns = new double[nPeriods];
        double[] benchmarkReturns = new double[nPeriods];
        
        for (int i = 0; i < nPeriods; i++) {
            benchmarkReturns[i] = 0.0006 + 0.012 * random.nextGaussian();
            portfolioReturns[i] = 0.0008 + 1.2 * benchmarkReturns[i] + 0.008 * random.nextGaussian();
        }
        
        System.out.printf("Portfolio Setup:\\n");
        System.out.printf("  Time periods: %d\\n", nPeriods);
        System.out.printf("  Average daily return: %.6f\\n", 
            Arrays.stream(portfolioReturns).average().orElse(0.0));
        System.out.printf("  Daily volatility: %.6f\\n\\n", 
            calculateVolatility(portfolioReturns));
        
        // Initialize risk manager
        PortfolioRiskManager riskManager = new PortfolioRiskManager(portfolioReturns, benchmarkReturns);
        
        // Calculate comprehensive risk metrics
        RiskMetrics riskMetrics = riskManager.calculateRiskMetrics(0.0002);
        
        System.out.println("=== Risk Metrics ===");
        System.out.printf("VaR (95%%):           %.4f\\n", riskMetrics.var95);
        System.out.printf("VaR (99%%):           %.4f\\n", riskMetrics.var99);
        System.out.printf("CVaR (95%%):          %.4f\\n", riskMetrics.cvar95);
        System.out.printf("CVaR (99%%):          %.4f\\n", riskMetrics.cvar99);
        System.out.printf("Maximum Drawdown:    %.4f\\n", riskMetrics.maxDrawdown);
        System.out.printf("Volatility:          %.4f\\n", riskMetrics.volatility);
        System.out.printf("Sharpe Ratio:        %.4f\\n", riskMetrics.sharpeRatio);
        System.out.printf("Sortino Ratio:       %.4f\\n", riskMetrics.sortinoRatio);
        System.out.printf("Calmar Ratio:        %.4f\\n", riskMetrics.calmarRatio);
        System.out.printf("Beta:                %.4f\\n", riskMetrics.beta);
        System.out.printf("Tracking Error:      %.4f\\n\\n", riskMetrics.trackingError);
        
        // VaR method comparison
        System.out.println("=== VaR Method Comparison ===");
        String[] methods = {"historical", "parametric", "monte_carlo"};
        double[] confidenceLevels = {0.95, 0.99};
        
        System.out.printf("%-15s %-12s %-12s\\n", "Method", "95% VaR", "99% VaR");
        System.out.println("-".repeat(40));
        
        for (String method : methods) {
            double var95, var99;
            
            if (method.equals("historical")) {
                var95 = riskManager.calculateHistoricalVaR(0.95);
                var99 = riskManager.calculateHistoricalVaR(0.99);
            } else if (method.equals("parametric")) {
                var95 = riskManager.calculateParametricVaR(0.95);
                var99 = riskManager.calculateParametricVaR(0.99);
            } else {
                var95 = riskManager.calculateMonteCarloVaR(0.95, 10000);
                var99 = riskManager.calculateMonteCarloVaR(0.99, 10000);
            }
            
            System.out.printf("%-15s %-12.6f %-12.6f\\n", method, var95, var99);
        }
        
        System.out.println();
        
        // Stress testing
        System.out.println("=== Stress Testing ===");
        
        Map<String, Map<String, Double>> stressScenarios = new HashMap<>();
        
        Map<String, Double> marketCrash = new HashMap<>();
        marketCrash.put("mean_shock", -0.02);
        marketCrash.put("vol_shock", 2.0);
        stressScenarios.put("Market Crash", marketCrash);
        
        Map<String, Double> volSpike = new HashMap<>();
        volSpike.put("mean_shock", 0.0);
        volSpike.put("vol_shock", 1.5);
        stressScenarios.put("Volatility Spike", volSpike);
        
        Map<String, Double> deflation = new HashMap<>();
        deflation.put("mean_shock", -0.005);
        deflation.put("vol_shock", 0.8);
        stressScenarios.put("Deflationary", deflation);
        
        Map<String, Map<String, Double>> stressResults = riskManager.stressTest(stressScenarios);
        
        System.out.printf("%-15s %-12s %-12s %-12s\\n", "Scenario", "VaR (95%)", "CVaR (95%)", "Max DD");
        System.out.println("-".repeat(60));
        
        for (Map.Entry<String, Map<String, Double>> entry : stressResults.entrySet()) {
            String scenario = entry.getKey();
            Map<String, Double> results = entry.getValue();
            
            System.out.printf("%-15s %-12.6f %-12.6f %-12.6f\\n",
                scenario,
                results.get("var_95"),
                results.get("cvar_95"),
                results.get("max_drawdown"));
        }
    }
    
    private static double calculateVolatility(double[] returns) {
        double mean = Arrays.stream(returns).average().orElse(0.0);
        double sumSquaredDeviations = Arrays.stream(returns)
            .map(r -> Math.pow(r - mean, 2))
            .sum();
        return Math.sqrt(sumSquaredDeviations / (returns.length - 1));
    }
}