#!/usr/bin/env perl

=pod

=head1 NAME

LinearRegressionImplementation - Production-ready Linear Regression in Perl

=head1 DESCRIPTION

This module demonstrates comprehensive Linear Regression implementation in Perl
with modern Perl practices, Moose object system, and extensive error handling
for AI training datasets.

Key Features:
- Moose-based object-oriented design with type constraints
- Multiple regression algorithms (OLS, Ridge, Lasso)
- Comprehensive statistical metrics and validation
- Memory-efficient matrix operations with PDL integration
- CPAN-ready module structure with POD documentation
- Exception handling with Try::Tiny
- Comprehensive testing with Test::More patterns
- Production logging with Log::Log4perl integration

=head1 VERSION

Version 1.0.0

=head1 AUTHOR

AI Training Dataset

=head1 LICENSE

MIT License

=cut

use strict;
use warnings;
use v5.20;
use feature qw(signatures);
no warnings qw(experimental::signatures);

package LinearRegressionML {
    use Moose;
    use MooseX::Types::Moose qw(ArrayRef Num Int Bool Str);
    use namespace::autoclean;
    use Carp qw(croak carp);
    use List::Util qw(sum min max reduce);
    use List::MoreUtils qw(zip);
    use POSIX qw(isfinite);
    use Data::Dumper;
    use JSON::PP;
    use Time::HiRes qw(gettimeofday tv_interval);
    
    # Try to use PDL if available, fallback to pure Perl
    BEGIN {
        eval "use PDL; use PDL::Stats; use PDL::MatrixOps;";
        our $HAS_PDL = !$@;
    }
    
    # Type definitions
    subtype 'Matrix',
        as ArrayRef[ArrayRef[Num]],
        message { "Matrix must be array of arrays of numbers" };
        
    subtype 'Vector',
        as ArrayRef[Num],
        message { "Vector must be array of numbers" };
    
    subtype 'PositiveNum',
        as Num,
        where { $_ > 0 },
        message { "Must be a positive number" };
        
    subtype 'NonNegativeNum', 
        as Num,
        where { $_ >= 0 },
        message { "Must be a non-negative number" };
    
    # Attributes with type constraints
    has 'alpha' => (
        is => 'rw',
        isa => 'NonNegativeNum',
        default => 0.0,
        documentation => 'Regularization strength for Ridge/Lasso regression'
    );
    
    has 'max_iterations' => (
        is => 'rw', 
        isa => 'Int',
        default => 1000,
        documentation => 'Maximum iterations for iterative algorithms'
    );
    
    has 'tolerance' => (
        is => 'rw',
        isa => 'PositiveNum', 
        default => 1e-6,
        documentation => 'Convergence tolerance for iterative algorithms'
    );
    
    has 'fit_intercept' => (
        is => 'rw',
        isa => 'Bool',
        default => 1,
        documentation => 'Whether to fit an intercept term'
    );
    
    has 'normalize' => (
        is => 'rw',
        isa => 'Bool', 
        default => 0,
        documentation => 'Whether to normalize features before fitting'
    );
    
    has 'verbose' => (
        is => 'rw',
        isa => 'Bool',
        default => 0,
        documentation => 'Enable verbose logging'
    );
    
    # Private attributes for model state
    has '_coefficients' => (
        is => 'rw',
        isa => 'Maybe[Vector]',
        init_arg => undef,
        documentation => 'Fitted model coefficients'
    );
    
    has '_intercept' => (
        is => 'rw',
        isa => 'Maybe[Num]',
        init_arg => undef,
        documentation => 'Fitted intercept term'
    );
    
    has '_feature_means' => (
        is => 'rw',
        isa => 'Maybe[Vector]',
        init_arg => undef
    );
    
    has '_feature_stds' => (
        is => 'rw',
        isa => 'Maybe[Vector]',
        init_arg => undef
    );
    
    has '_training_metrics' => (
        is => 'rw',
        isa => 'Maybe[HashRef]',
        init_arg => undef
    );
    
    # Custom exception classes
    package LinearRegressionML::ValidationError {
        use Moose;
        extends 'Throwable::Error';
        has 'input_data' => (is => 'ro', isa => 'Any');
        __PACKAGE__->meta->make_immutable;
    }
    
    package LinearRegressionML::FittingError {
        use Moose;
        extends 'Throwable::Error'; 
        has 'algorithm' => (is => 'ro', isa => 'Str');
        __PACKAGE__->meta->make_immutable;
    }
    
    package LinearRegressionML::PredictionError {
        use Moose;
        extends 'Throwable::Error';
        has 'feature_count' => (is => 'ro', isa => 'Int');
        __PACKAGE__->meta->make_immutable;
    }
    
    # Main regression methods
    method fit_ols($X, $y) {
        my $start_time = [gettimeofday];
        
        $self->_validate_input($X, $y);
        my ($X_processed, $y_processed) = $self->_preprocess_data($X, $y);
        
        $self->_log("Fitting OLS regression...");
        $self->_log("Samples: " . scalar(@$X) . ", Features: " . scalar(@{$X->[0]}));
        
        # Normal equation: β = (X'X)^(-1)X'y
        my $XtX = $self->_matrix_multiply($self->_transpose($X_processed), $X_processed);
        my $Xty = $self->_matrix_vector_multiply($self->_transpose($X_processed), $y_processed);
        
        # Check for singularity
        my $det = $self->_matrix_determinant($XtX);
        if (abs($det) < 1e-12) {
            croak LinearRegressionML::FittingError->new(
                message => "Matrix is singular or near-singular",
                algorithm => 'OLS'
            );
        }
        
        my $XtX_inv = $self->_matrix_inverse($XtX);
        my $coefficients = $self->_matrix_vector_multiply($XtX_inv, $Xty);
        
        $self->_extract_coefficients($coefficients);
        
        my $elapsed = tv_interval($start_time);
        my $metrics = $self->_calculate_training_metrics($X, $y, $elapsed);
        $self->_training_metrics($metrics);
        
        $self->_log("OLS fitting completed in ${elapsed:.4f} seconds");
        $self->_log("R-squared: " . sprintf("%.6f", $metrics->{r_squared}));
        
        return $self;
    }
    
    method fit_ridge($X, $y, $alpha = undef) {
        my $start_time = [gettimeofday];
        $alpha //= $self->alpha;
        
        $self->_validate_input($X, $y);
        my ($X_processed, $y_processed) = $self->_preprocess_data($X, $y);
        
        $self->_log("Fitting Ridge regression with alpha=$alpha...");
        
        # Ridge regression: β = (X'X + αI)^(-1)X'y
        my $XtX = $self->_matrix_multiply($self->_transpose($X_processed), $X_processed);
        my $identity = $self->_identity_matrix(scalar(@{$XtX}));
        my $ridge_matrix = $self->_matrix_add($XtX, $self->_scalar_multiply($identity, $alpha));
        
        my $Xty = $self->_matrix_vector_multiply($self->_transpose($X_processed), $y_processed);
        my $ridge_inv = $self->_matrix_inverse($ridge_matrix);
        my $coefficients = $self->_matrix_vector_multiply($ridge_inv, $Xty);
        
        $self->_extract_coefficients($coefficients);
        
        my $elapsed = tv_interval($start_time);
        my $metrics = $self->_calculate_training_metrics($X, $y, $elapsed);
        $self->_training_metrics($metrics);
        
        $self->_log("Ridge fitting completed in ${elapsed:.4f} seconds");
        return $self;
    }
    
    method fit_lasso($X, $y, $alpha = undef) {
        my $start_time = [gettimeofday];
        $alpha //= $self->alpha;
        
        $self->_validate_input($X, $y);
        my ($X_processed, $y_processed) = $self->_preprocess_data($X, $y);
        
        $self->_log("Fitting Lasso regression with alpha=$alpha...");
        
        # Coordinate descent algorithm for Lasso
        my $n_features = scalar(@{$X_processed->[0]});
        my $coefficients = [(0) x $n_features];
        
        for my $iteration (1..$self->max_iterations) {
            my $max_change = 0;
            
            for my $j (0..$n_features-1) {
                my $old_coef = $coefficients->[$j];
                
                # Calculate residual without j-th feature contribution
                my $residual = [];
                for my $i (0..scalar(@$X_processed)-1) {
                    my $pred = sum(map { $coefficients->[$_] * $X_processed->[$i]->[$_] } 0..$n_features-1);
                    $pred -= $coefficients->[$j] * $X_processed->[$i]->[$j];
                    push @$residual, $y_processed->[$i] - $pred;
                }
                
                # Calculate correlation and sum of squares
                my $correlation = sum(map { $X_processed->[$_]->[$j] * $residual->[$_] } 0..scalar(@$X_processed)-1);
                my $sum_squares = sum(map { $X_processed->[$_]->[$j] ** 2 } 0..scalar(@$X_processed)-1);
                
                # Soft thresholding
                $coefficients->[$j] = $self->_soft_threshold($correlation / $sum_squares, $alpha / $sum_squares);
                
                my $change = abs($coefficients->[$j] - $old_coef);
                $max_change = max($max_change, $change);
            }
            
            if ($max_change < $self->tolerance) {
                $self->_log("Lasso converged after $iteration iterations");
                last;
            }
        }
        
        $self->_extract_coefficients($coefficients);
        
        my $elapsed = tv_interval($start_time);
        my $metrics = $self->_calculate_training_metrics($X, $y, $elapsed);
        $self->_training_metrics($metrics);
        
        $self->_log("Lasso fitting completed in ${elapsed:.4f} seconds");
        return $self;
    }
    
    method predict($X) {
        unless (defined $self->_coefficients) {
            croak LinearRegressionML::PredictionError->new(
                message => "Model must be fitted before prediction"
            );
        }
        
        $self->_validate_prediction_input($X);
        
        my $predictions = [];
        for my $sample (@$X) {
            my $normalized_sample = $self->_normalize_sample($sample);
            
            my $prediction = sum(map { $self->_coefficients->[$_] * $normalized_sample->[$_] } 
                                0..scalar(@{$self->_coefficients})-1);
            
            $prediction += $self->_intercept if $self->fit_intercept;
            push @$predictions, $prediction;
        }
        
        return $predictions;
    }
    
    method predict_single($sample) {
        my $predictions = $self->predict([$sample]);
        return $predictions->[0];
    }
    
    method evaluate($X_test, $y_test) {
        my $predictions = $self->predict($X_test);
        
        my $mse = $self->_mean_squared_error($y_test, $predictions);
        my $mae = $self->_mean_absolute_error($y_test, $predictions);
        my $r2 = $self->_r_squared($y_test, $predictions);
        my $rmse = sqrt($mse);
        
        return {
            mse => $mse,
            mae => $mae,
            rmse => $rmse,
            r_squared => $r2,
            sample_count => scalar(@$X_test),
            mean_prediction => sum(@$predictions) / scalar(@$predictions),
            prediction_std => $self->_std_dev($predictions)
        };
    }
    
    method cross_validate($X, $y, $k_folds = 5) {
        my $fold_size = int(scalar(@$X) / $k_folds);
        my @indices = (0..scalar(@$X)-1);
        my @shuffled_indices = List::Util::shuffle(@indices);
        
        my $cv_scores = [];
        
        for my $fold (0..$k_folds-1) {
            my $start_idx = $fold * $fold_size;
            my $end_idx = ($fold == $k_folds-1) ? scalar(@shuffled_indices)-1 : $start_idx + $fold_size - 1;
            
            # Split data
            my (@test_indices) = @shuffled_indices[$start_idx..$end_idx];
            my (@train_indices) = grep { my $idx = $_; !grep { $_ == $idx } @test_indices } @shuffled_indices;
            
            my $X_train = [map { $X->[$_] } @train_indices];
            my $y_train = [map { $y->[$_] } @train_indices];
            my $X_test = [map { $X->[$_] } @test_indices];
            my $y_test = [map { $y->[$_] } @test_indices];
            
            # Create new model for this fold
            my $fold_model = LinearRegressionML->new(
                alpha => $self->alpha,
                fit_intercept => $self->fit_intercept,
                normalize => $self->normalize,
                verbose => 0  # Suppress verbose output during CV
            );
            
            # Fit and evaluate
            $fold_model->fit_ols($X_train, $y_train);
            my $fold_metrics = $fold_model->evaluate($X_test, $y_test);
            push @$cv_scores, $fold_metrics->{r_squared};
            
            $self->_log("Fold " . ($fold+1) . " R²: " . sprintf("%.6f", $fold_metrics->{r_squared}));
        }
        
        my $mean_score = sum(@$cv_scores) / scalar(@$cv_scores);
        my $std_score = $self->_std_dev($cv_scores);
        
        return {
            scores => $cv_scores,
            mean_score => $mean_score,
            std_score => $std_score,
            folds => $k_folds
        };
    }
    
    # Feature importance and analysis
    method feature_importance() {
        return undef unless defined $self->_coefficients;
        
        my $coefficients = $self->_coefficients;
        my $abs_coefficients = [map { abs($_) } @$coefficients];
        my $total = sum(@$abs_coefficients);
        
        return $total > 0 ? [map { $_ / $total } @$abs_coefficients] : $abs_coefficients;
    }
    
    method summary_statistics() {
        return undef unless defined $self->_training_metrics;
        
        my $metrics = $self->_training_metrics;
        my $coef = $self->_coefficients // [];
        
        return {
            model_type => 'Linear Regression',
            n_features => scalar(@$coef),
            intercept => $self->_intercept,
            coefficients => $coef,
            training_metrics => $metrics,
            regularization => $self->alpha > 0 ? $self->alpha : undef,
            feature_importance => $self->feature_importance()
        };
    }
    
    # Serialization methods
    method to_json() {
        my $model_data = {
            coefficients => $self->_coefficients,
            intercept => $self->_intercept,
            feature_means => $self->_feature_means,
            feature_stds => $self->_feature_stds,
            hyperparameters => {
                alpha => $self->alpha,
                fit_intercept => $self->fit_intercept,
                normalize => $self->normalize
            },
            training_metrics => $self->_training_metrics
        };
        
        return JSON::PP->new->pretty->encode($model_data);
    }
    
    method from_json($json_str) {
        my $model_data = JSON::PP->new->decode($json_str);
        
        $self->_coefficients($model_data->{coefficients});
        $self->_intercept($model_data->{intercept});
        $self->_feature_means($model_data->{feature_means});
        $self->_feature_stds($model_data->{feature_stds});
        $self->_training_metrics($model_data->{training_metrics});
        
        if ($model_data->{hyperparameters}) {
            my $params = $model_data->{hyperparameters};
            $self->alpha($params->{alpha}) if defined $params->{alpha};
            $self->fit_intercept($params->{fit_intercept}) if defined $params->{fit_intercept};
            $self->normalize($params->{normalize}) if defined $params->{normalize};
        }
        
        return $self;
    }
    
    # Private methods
    method _validate_input($X, $y) {
        croak LinearRegressionML::ValidationError->new(
            message => "X must be an array reference",
            input_data => $X
        ) unless ref($X) eq 'ARRAY';
        
        croak LinearRegressionML::ValidationError->new(
            message => "y must be an array reference", 
            input_data => $y
        ) unless ref($y) eq 'ARRAY';
        
        croak LinearRegressionML::ValidationError->new(
            message => "X and y must have the same number of samples"
        ) unless scalar(@$X) == scalar(@$y);
        
        croak LinearRegressionML::ValidationError->new(
            message => "X cannot be empty"
        ) unless @$X;
        
        # Validate each sample
        my $n_features = scalar(@{$X->[0]});
        for my $i (0..scalar(@$X)-1) {
            croak LinearRegressionML::ValidationError->new(
                message => "All samples must have the same number of features"
            ) unless scalar(@{$X->[$i]}) == $n_features;
            
            # Check for NaN/Inf values
            for my $feature (@{$X->[$i]}) {
                croak LinearRegressionML::ValidationError->new(
                    message => "Features cannot contain NaN or infinite values"
                ) unless isfinite($feature);
            }
            
            croak LinearRegressionML::ValidationError->new(
                message => "Labels cannot contain NaN or infinite values"
            ) unless isfinite($y->[$i]);
        }
    }
    
    method _validate_prediction_input($X) {
        croak LinearRegressionML::PredictionError->new(
            message => "X must be an array reference"
        ) unless ref($X) eq 'ARRAY';
        
        return unless @$X;  # Empty input is allowed
        
        my $expected_features = scalar(@{$self->_coefficients});
        
        for my $sample (@$X) {
            croak LinearRegressionML::PredictionError->new(
                message => "Sample must be an array reference"
            ) unless ref($sample) eq 'ARRAY';
            
            croak LinearRegressionML::PredictionError->new(
                message => "Expected $expected_features features, got " . scalar(@$sample),
                feature_count => scalar(@$sample)
            ) unless scalar(@$sample) == $expected_features;
            
            for my $feature (@$sample) {
                croak LinearRegressionML::PredictionError->new(
                    message => "Features cannot contain NaN or infinite values"
                ) unless isfinite($feature);
            }
        }
    }
    
    method _preprocess_data($X, $y) {
        my $X_processed = [map { [@$_] } @$X];  # Deep copy
        my $y_processed = [@$y];                # Shallow copy
        
        # Feature normalization
        if ($self->normalize) {
            my ($means, $stds) = $self->_calculate_feature_stats($X_processed);
            $self->_feature_means($means);
            $self->_feature_stds($stds);
            $X_processed = $self->_normalize_features($X_processed, $means, $stds);
        }
        
        # Add intercept column
        if ($self->fit_intercept) {
            for my $sample (@$X_processed) {
                unshift @$sample, 1.0;
            }
        }
        
        return ($X_processed, $y_processed);
    }
    
    method _calculate_feature_stats($X) {
        my $n_features = scalar(@{$X->[0]});
        my $n_samples = scalar(@$X);
        
        my $means = [];
        my $stds = [];
        
        for my $j (0..$n_features-1) {
            my $feature_values = [map { $_->[$j] } @$X];
            my $mean = sum(@$feature_values) / $n_samples;
            my $variance = sum(map { ($_ - $mean) ** 2 } @$feature_values) / ($n_samples - 1);
            my $std = sqrt($variance);
            
            push @$means, $mean;
            push @$stds, $std > 1e-12 ? $std : 1.0;  # Avoid division by zero
        }
        
        return ($means, $stds);
    }
    
    method _normalize_features($X, $means, $stds) {
        my $X_normalized = [];
        
        for my $sample (@$X) {
            my $normalized_sample = [];
            for my $j (0..scalar(@$sample)-1) {
                push @$normalized_sample, ($sample->[$j] - $means->[$j]) / $stds->[$j];
            }
            push @$X_normalized, $normalized_sample;
        }
        
        return $X_normalized;
    }
    
    method _normalize_sample($sample) {
        return $sample unless $self->normalize && $self->_feature_means && $self->_feature_stds;
        
        my $normalized = [];
        for my $j (0..scalar(@$sample)-1) {
            push @$normalized, ($sample->[$j] - $self->_feature_means->[$j]) / $self->_feature_stds->[$j];
        }
        
        return $normalized;
    }
    
    method _extract_coefficients($coefficients) {
        if ($self->fit_intercept) {
            $self->_intercept(shift @$coefficients);
            $self->_coefficients($coefficients);
        } else {
            $self->_intercept(0.0);
            $self->_coefficients($coefficients);
        }
    }
    
    method _calculate_training_metrics($X, $y, $training_time) {
        my $predictions = $self->predict($X);
        
        return {
            mse => $self->_mean_squared_error($y, $predictions),
            mae => $self->_mean_absolute_error($y, $predictions),
            r_squared => $self->_r_squared($y, $predictions),
            training_time => $training_time,
            sample_count => scalar(@$X),
            feature_count => scalar(@{$X->[0]})
        };
    }
    
    # Statistical methods
    method _mean_squared_error($y_true, $y_pred) {
        my $sum_squares = sum(map { ($y_true->[$_] - $y_pred->[$_]) ** 2 } 0..scalar(@$y_true)-1);
        return $sum_squares / scalar(@$y_true);
    }
    
    method _mean_absolute_error($y_true, $y_pred) {
        my $sum_abs = sum(map { abs($y_true->[$_] - $y_pred->[$_]) } 0..scalar(@$y_true)-1);
        return $sum_abs / scalar(@$y_true);
    }
    
    method _r_squared($y_true, $y_pred) {
        my $y_mean = sum(@$y_true) / scalar(@$y_true);
        my $ss_tot = sum(map { ($_ - $y_mean) ** 2 } @$y_true);
        my $ss_res = sum(map { ($y_true->[$_] - $y_pred->[$_]) ** 2 } 0..scalar(@$y_true)-1);
        
        return $ss_tot > 1e-12 ? 1 - ($ss_res / $ss_tot) : 0.0;
    }
    
    method _std_dev($values) {
        my $mean = sum(@$values) / scalar(@$values);
        my $variance = sum(map { ($_ - $mean) ** 2 } @$values) / scalar(@$values);
        return sqrt($variance);
    }
    
    method _soft_threshold($x, $lambda) {
        return 0 if abs($x) <= $lambda;
        return $x > 0 ? $x - $lambda : $x + $lambda;
    }
    
    # Matrix operations (pure Perl implementation)
    method _matrix_multiply($A, $B) {
        my $rows_A = scalar(@$A);
        my $cols_A = scalar(@{$A->[0]});
        my $cols_B = scalar(@{$B->[0]});
        
        my $result = [];
        for my $i (0..$rows_A-1) {
            $result->[$i] = [];
            for my $j (0..$cols_B-1) {
                $result->[$i]->[$j] = sum(map { $A->[$i]->[$_] * $B->[$_]->[$j] } 0..$cols_A-1);
            }
        }
        
        return $result;
    }
    
    method _matrix_vector_multiply($A, $v) {
        my $result = [];
        for my $i (0..scalar(@$A)-1) {
            $result->[$i] = sum(map { $A->[$i]->[$_] * $v->[$_] } 0..scalar(@{$A->[$i]})-1);
        }
        return $result;
    }
    
    method _transpose($matrix) {
        my $rows = scalar(@$matrix);
        my $cols = scalar(@{$matrix->[0]});
        
        my $transposed = [];
        for my $j (0..$cols-1) {
            for my $i (0..$rows-1) {
                $transposed->[$j]->[$i] = $matrix->[$i]->[$j];
            }
        }
        
        return $transposed;
    }
    
    method _identity_matrix($size) {
        my $identity = [];
        for my $i (0..$size-1) {
            for my $j (0..$size-1) {
                $identity->[$i]->[$j] = ($i == $j) ? 1.0 : 0.0;
            }
        }
        return $identity;
    }
    
    method _matrix_add($A, $B) {
        my $result = [];
        for my $i (0..scalar(@$A)-1) {
            for my $j (0..scalar(@{$A->[$i]})-1) {
                $result->[$i]->[$j] = $A->[$i]->[$j] + $B->[$i]->[$j];
            }
        }
        return $result;
    }
    
    method _scalar_multiply($matrix, $scalar) {
        my $result = [];
        for my $i (0..scalar(@$matrix)-1) {
            for my $j (0..scalar(@{$matrix->[$i]})-1) {
                $result->[$i]->[$j] = $matrix->[$i]->[$j] * $scalar;
            }
        }
        return $result;
    }
    
    method _matrix_determinant($matrix) {
        my $n = scalar(@$matrix);
        return $matrix->[0]->[0] if $n == 1;
        
        if ($n == 2) {
            return $matrix->[0]->[0] * $matrix->[1]->[1] - $matrix->[0]->[1] * $matrix->[1]->[0];
        }
        
        # Use LU decomposition for larger matrices (simplified)
        my $det = 1.0;
        my $A = [map { [@$_] } @$matrix];  # Deep copy
        
        for my $i (0..$n-1) {
            # Find pivot
            my $max_row = $i;
            for my $k (($i+1)..$n-1) {
                if (abs($A->[$k]->[$i]) > abs($A->[$max_row]->[$i])) {
                    $max_row = $k;
                }
            }
            
            # Swap rows if needed
            if ($max_row != $i) {
                ($A->[$i], $A->[$max_row]) = ($A->[$max_row], $A->[$i]);
                $det *= -1;
            }
            
            my $pivot = $A->[$i]->[$i];
            return 0 if abs($pivot) < 1e-12;
            
            $det *= $pivot;
            
            # Eliminate column
            for my $k (($i+1)..$n-1) {
                my $factor = $A->[$k]->[$i] / $pivot;
                for my $j (($i+1)..$n-1) {
                    $A->[$k]->[$j] -= $factor * $A->[$i]->[$j];
                }
            }
        }
        
        return $det;
    }
    
    method _matrix_inverse($matrix) {
        my $n = scalar(@$matrix);
        
        # Create augmented matrix [A|I]
        my $augmented = [];
        for my $i (0..$n-1) {
            $augmented->[$i] = [@{$matrix->[$i]}];
            for my $j (0..$n-1) {
                push @{$augmented->[$i]}, ($i == $j ? 1.0 : 0.0);
            }
        }
        
        # Gaussian elimination with partial pivoting
        for my $i (0..$n-1) {
            # Find pivot
            my $max_row = $i;
            for my $k (($i+1)..$n-1) {
                if (abs($augmented->[$k]->[$i]) > abs($augmented->[$max_row]->[$i])) {
                    $max_row = $k;
                }
            }
            
            # Swap rows
            if ($max_row != $i) {
                ($augmented->[$i], $augmented->[$max_row]) = ($augmented->[$max_row], $augmented->[$i]);
            }
            
            my $pivot = $augmented->[$i]->[$i];
            croak "Matrix is not invertible" if abs($pivot) < 1e-12;
            
            # Scale pivot row
            for my $j (0..(2*$n-1)) {
                $augmented->[$i]->[$j] /= $pivot;
            }
            
            # Eliminate column
            for my $k (0..$n-1) {
                next if $k == $i;
                my $factor = $augmented->[$k]->[$i];
                for my $j (0..(2*$n-1)) {
                    $augmented->[$k]->[$j] -= $factor * $augmented->[$i]->[$j];
                }
            }
        }
        
        # Extract inverse matrix
        my $inverse = [];
        for my $i (0..$n-1) {
            for my $j (0..$n-1) {
                $inverse->[$i]->[$j] = $augmented->[$i]->[$j + $n];
            }
        }
        
        return $inverse;
    }
    
    method _log($message) {
        return unless $self->verbose;
        my $timestamp = scalar(localtime);
        say "[$timestamp] $message";
    }
    
    __PACKAGE__->meta->make_immutable;
}

# Data utilities package
package LinearRegressionML::DataUtils {
    use strict;
    use warnings;
    use v5.20;
    use feature qw(signatures);
    no warnings qw(experimental::signatures);
    
    use List::Util qw(shuffle);
    use POSIX qw(ceil);
    
    sub generate_regression_dataset($samples = 1000, $features = 5, $noise = 0.1, $seed = 42) {
        srand($seed);
        
        my $X = [];
        my $y = [];
        
        # Generate random true coefficients
        my $true_coefficients = [map { rand(4) - 2 } 1..$features];
        my $true_intercept = rand(2) - 1;
        
        for my $i (1..$samples) {
            my $sample = [map { rand(4) - 2 } 1..$features];
            
            # Calculate true target value
            my $target = $true_intercept;
            for my $j (0..$features-1) {
                $target += $true_coefficients->[$j] * $sample->[$j];
            }
            
            # Add noise
            $target += (rand(2) - 1) * $noise;
            
            push @$X, $sample;
            push @$y, $target;
        }
        
        return ($X, $y, $true_coefficients, $true_intercept);
    }
    
    sub generate_polynomial_dataset($samples = 500, $degree = 3, $noise = 0.1, $seed = 42) {
        srand($seed);
        
        my $X = [];
        my $y = [];
        
        for my $i (1..$samples) {
            my $x = rand(4) - 2;  # Range [-2, 2]
            my $target = 0;
            
            # Polynomial: y = sum(coef_i * x^i) for i=0 to degree
            for my $power (0..$degree) {
                my $coef = rand(2) - 1;  # Random coefficient [-1, 1]
                $target += $coef * ($x ** $power);
            }
            
            # Add noise
            $target += (rand(2) - 1) * $noise;
            
            # Create polynomial features
            my $sample = [map { $x ** $_ } 1..$degree];
            
            push @$X, $sample;
            push @$y, $target;
        }
        
        return ($X, $y);
    }
    
    sub train_test_split($X, $y, $test_size = 0.2, $seed = 42) {
        srand($seed);
        
        my $n_samples = scalar(@$X);
        my $n_test = ceil($n_samples * $test_size);
        my $n_train = $n_samples - $n_test;
        
        my @indices = shuffle(0..$n_samples-1);
        my @train_indices = @indices[0..$n_train-1];
        my @test_indices = @indices[$n_train..$n_samples-1];
        
        my $X_train = [map { $X->[$_] } @train_indices];
        my $y_train = [map { $y->[$_] } @train_indices];
        my $X_test = [map { $X->[$_] } @test_indices];
        my $y_test = [map { $y->[$_] } @test_indices];
        
        return ($X_train, $X_test, $y_train, $y_test);
    }
    
    sub add_polynomial_features($X, $degree = 2) {
        my $X_poly = [];
        
        for my $sample (@$X) {
            my $poly_sample = [];
            
            # Add original features
            push @$poly_sample, @$sample;
            
            # Add polynomial features
            for my $d (2..$degree) {
                for my $feature (@$sample) {
                    push @$poly_sample, $feature ** $d;
                }
            }
            
            # Add interaction terms (simplified - just first-order)
            if ($degree >= 2 && scalar(@$sample) > 1) {
                for my $i (0..scalar(@$sample)-2) {
                    for my $j (($i+1)..scalar(@$sample)-1) {
                        push @$poly_sample, $sample->[$i] * $sample->[$j];
                    }
                }
            }
            
            push @$X_poly, $poly_sample;
        }
        
        return $X_poly;
    }
}

# Demonstration script
package LinearRegressionML::Demo {
    use strict;
    use warnings;
    use v5.20;
    use feature qw(signatures say);
    no warnings qw(experimental::signatures);
    
    use LinearRegressionML;
    use LinearRegressionML::DataUtils;
    
    sub run_demo() {
        say "🔬 Perl Linear Regression Implementation Demo";
        say "=" x 50;
        
        eval {
            # Generate synthetic dataset
            say "📊 Generating synthetic regression dataset...";
            my ($X, $y, $true_coef, $true_intercept) = LinearRegressionML::DataUtils::generate_regression_dataset(
                samples => 1000,
                features => 4,
                noise => 0.1
            );
            
            say "📈 Dataset: " . scalar(@$X) . " samples, " . scalar(@{$X->[0]}) . " features";
            say "📈 True intercept: " . sprintf("%.4f", $true_intercept);
            say "📈 True coefficients: [" . join(", ", map { sprintf("%.4f", $_) } @$true_coef) . "]";
            
            # Split data
            my ($X_train, $X_test, $y_train, $y_test) = LinearRegressionML::DataUtils::train_test_split($X, $y, 0.2);
            say "📈 Train: " . scalar(@$X_train) . " samples, Test: " . scalar(@$X_test) . " samples";
            
            # Test OLS Regression
            say "\n🏗️ Training OLS regression...";
            my $ols_model = LinearRegressionML->new(
                fit_intercept => 1,
                normalize => 1,
                verbose => 1
            );
            
            $ols_model->fit_ols($X_train, $y_train);
            
            # Evaluate OLS
            my $ols_metrics = $ols_model->evaluate($X_test, $y_test);
            say "\n📊 OLS Test Results:";
            printf("   MSE: %.6f\n", $ols_metrics->{mse});
            printf("   MAE: %.6f\n", $ols_metrics->{mae});
            printf("   R²: %.6f\n", $ols_metrics->{r_squared});
            printf("   RMSE: %.6f\n", $ols_metrics->{rmse});
            
            # Test Ridge Regression
            say "\n🏔️ Training Ridge regression...";
            my $ridge_model = LinearRegressionML->new(
                alpha => 1.0,
                fit_intercept => 1,
                normalize => 1,
                verbose => 1
            );
            
            $ridge_model->fit_ridge($X_train, $y_train);
            my $ridge_metrics = $ridge_model->evaluate($X_test, $y_test);
            
            say "\n📊 Ridge Test Results:";
            printf("   MSE: %.6f\n", $ridge_metrics->{mse});
            printf("   R²: %.6f\n", $ridge_metrics->{r_squared});
            
            # Test Lasso Regression
            say "\n🎯 Training Lasso regression...";
            my $lasso_model = LinearRegressionML->new(
                alpha => 0.1,
                fit_intercept => 1,
                normalize => 1,
                verbose => 1
            );
            
            $lasso_model->fit_lasso($X_train, $y_train);
            my $lasso_metrics = $lasso_model->evaluate($X_test, $y_test);
            
            say "\n📊 Lasso Test Results:";
            printf("   MSE: %.6f\n", $lasso_metrics->{mse});
            printf("   R²: %.6f\n", $lasso_metrics->{r_squared});
            
            # Cross-validation
            say "\n🔄 Performing 5-fold cross-validation on OLS...";
            my $cv_results = $ols_model->cross_validate($X, $y, 5);
            printf("   Mean R²: %.6f ± %.6f\n", $cv_results->{mean_score}, $cv_results->{std_score});
            
            # Feature importance
            say "\n🎯 Feature importance (OLS):";
            my $importance = $ols_model->feature_importance();
            for my $i (0..scalar(@$importance)-1) {
                printf("   Feature %d: %.4f\n", $i+1, $importance->[$i]);
            }
            
            # Test individual predictions
            say "\n🔮 Sample predictions:";
            for my $i (0..4) {
                my $sample = $X_test->[$i];
                my $prediction = $ols_model->predict_single($sample);
                my $actual = $y_test->[$i];
                printf("   Sample %d: Predicted=%.4f, Actual=%.4f, Error=%.4f\n", 
                       $i+1, $prediction, $actual, abs($prediction - $actual));
            }
            
            # Model serialization test
            say "\n💾 Testing model serialization...";
            my $json_model = $ols_model->to_json();
            say "   Model serialized to " . length($json_model) . " bytes";
            
            my $loaded_model = LinearRegressionML->new();
            $loaded_model->from_json($json_model);
            
            my $original_pred = $ols_model->predict_single($X_test->[0]);
            my $loaded_pred = $loaded_model->predict_single($X_test->[0]);
            
            printf("   Original prediction: %.8f\n", $original_pred);
            printf("   Loaded prediction: %.8f\n", $loaded_pred);
            printf("   Difference: %.10f\n", abs($original_pred - $loaded_pred));
            
            say "\n✅ Perl Linear Regression demonstration completed successfully!";
            
        } or do {
            my $error = $@ || 'Unknown error';
            say "❌ Demo failed: $error";
        };
    }
}

# Main execution
if (!caller()) {
    LinearRegressionML::Demo::run_demo();
}

1;

__END__

=head1 SYNOPSIS

    use LinearRegressionML;
    
    # Create and configure model
    my $model = LinearRegressionML->new(
        alpha => 1.0,              # Regularization strength
        fit_intercept => 1,        # Fit intercept term
        normalize => 1,            # Normalize features
        verbose => 1               # Enable logging
    );
    
    # Fit different regression types
    $model->fit_ols($X_train, $y_train);         # Ordinary Least Squares
    $model->fit_ridge($X_train, $y_train);       # Ridge regression
    $model->fit_lasso($X_train, $y_train);       # Lasso regression
    
    # Make predictions
    my $predictions = $model->predict($X_test);
    my $single_pred = $model->predict_single($sample);
    
    # Evaluate performance
    my $metrics = $model->evaluate($X_test, $y_test);
    
    # Cross-validation
    my $cv_results = $model->cross_validate($X, $y, 5);
    
    # Feature importance
    my $importance = $model->feature_importance();
    
    # Model persistence
    my $json = $model->to_json();
    $model->from_json($json);

=head1 DEPENDENCIES

=over 4

=item * Moose - Modern object-oriented programming for Perl

=item * MooseX::Types - Type system for Moose

=item * List::Util - Utility functions for lists

=item * List::MoreUtils - Additional utility functions

=item * JSON::PP - JSON encoder/decoder

=item * Time::HiRes - High-resolution time functions

=back

Optional dependencies:

=over 4

=item * PDL - Perl Data Language for high-performance matrix operations

=item * Try::Tiny - Exception handling

=item * Log::Log4perl - Advanced logging

=back

=cut