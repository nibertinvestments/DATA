#!/usr/bin/env perl

=pod

=head1 Advanced Perl Programming Examples

This module demonstrates intermediate to advanced Perl concepts including:
- Advanced regular expressions and text processing
- Object-oriented programming with Moose
- Functional programming techniques
- References, complex data structures, and closures
- Module development and package management
- System administration and file processing
- Web scraping and data extraction
- Performance optimization techniques

=cut

use strict;
use warnings;
use feature qw(say state signatures);
no warnings 'experimental::signatures';

use Data::Dumper;
use List::Util qw(reduce sum max min first);
use List::MoreUtils qw(uniq any all);
use JSON;
use Time::HiRes qw(time);
use Scalar::Util qw(blessed weaken);

# Advanced Regular Expressions and Text Processing
# ===============================================

package TextProcessor {
    use strict;
    use warnings;
    
    # Email validation with comprehensive regex
    sub validate_email($email) {
        my $email_regex = qr{
            ^
            [a-zA-Z0-9._%+-]+           # Username part
            @                           # @ symbol
            [a-zA-Z0-9.-]+              # Domain name
            \.                          # Dot
            [a-zA-Z]{2,}                # TLD
            $
        }x;
        
        return $email =~ $email_regex;
    }
    
    # Parse log files with named captures
    sub parse_apache_log($log_line) {
        my $log_regex = qr{
            ^
            (?<ip>\S+)                  # IP address
            \s+\S+\s+\S+\s+             # Remote logname and user
            \[(?<timestamp>[^\]]+)\]    # Timestamp
            \s+
            "(?<method>\S+)             # HTTP method
            \s+
            (?<path>\S+)                # Request path
            \s+
            (?<protocol>\S+)"           # HTTP protocol
            \s+
            (?<status>\d+)              # Status code
            \s+
            (?<size>\S+)                # Response size
        }x;
        
        if ($log_line =~ $log_regex) {
            return {
                ip => $+{ip},
                timestamp => $+{timestamp},
                method => $+{method},
                path => $+{path},
                protocol => $+{protocol},
                status => $+{status},
                size => $+{size} eq '-' ? 0 : $+{size}
            };
        }
        return undef;
    }
    
    # Advanced text substitution with callbacks
    sub process_markdown_links($text) {
        return $text =~ s{
            \[([^\]]+)\]        # Link text
            \(([^)]+)\)         # URL
        }{
            my ($text, $url) = ($1, $2);
            qq{<a href="$url">$text</a>}
        }gerx;
    }
    
    # Extract structured data using regex
    sub extract_csv_with_quoted_fields($csv_line) {
        my @fields;
        
        # Handle quoted fields with embedded commas
        while ($csv_line =~ m{
            (?:
                "([^"]*(?:""[^"]*)*)"   # Quoted field (with escaped quotes)
                |
                ([^,]*)                 # Unquoted field
            )
            (?:,|$)                     # Comma or end of line
        }gx) {
            if (defined $1) {
                # Quoted field - unescape quotes
                my $field = $1;
                $field =~ s/""/"/g;
                push @fields, $field;
            } else {
                # Unquoted field
                push @fields, $2;
            }
        }
        
        return \@fields;
    }
    
    # Word frequency analysis
    sub word_frequency($text) {
        # Convert to lowercase and extract words
        my @words = $text =~ /\b[a-zA-Z]+\b/g;
        @words = map { lc } @words;
        
        my %frequency;
        $frequency{$_}++ for @words;
        
        # Sort by frequency (descending)
        return [
            map { [$_, $frequency{$_}] }
            sort { $frequency{$b} <=> $frequency{$a} }
            keys %frequency
        ];
    }
}

# Object-Oriented Programming with Advanced Features
# =================================================

package BankAccount {
    use strict;
    use warnings;
    use Carp qw(croak);
    
    # Constructor with parameter validation
    sub new($class, %params) {
        my $account_number = $params{account_number} 
            or croak "Account number is required";
        my $initial_balance = $params{initial_balance} // 0;
        
        croak "Initial balance cannot be negative" 
            if $initial_balance < 0;
        
        my $self = {
            account_number => $account_number,
            balance => $initial_balance,
            transaction_history => [],
            created_at => time(),
        };
        
        return bless $self, $class;
    }
    
    # Getter methods
    sub account_number($self) { return $self->{account_number}; }
    sub balance($self) { return $self->{balance}; }
    sub transaction_history($self) { return @{$self->{transaction_history}}; }
    
    # Deposit method with validation
    sub deposit($self, $amount) {
        croak "Deposit amount must be positive" if $amount <= 0;
        
        $self->{balance} += $amount;
        $self->_add_transaction('deposit', $amount);
        
        return $self->{balance};
    }
    
    # Withdrawal method with validation
    sub withdraw($self, $amount) {
        croak "Withdrawal amount must be positive" if $amount <= 0;
        croak "Insufficient funds" if $amount > $self->{balance};
        
        $self->{balance} -= $amount;
        $self->_add_transaction('withdrawal', $amount);
        
        return $self->{balance};
    }
    
    # Transfer method
    sub transfer($self, $target_account, $amount) {
        croak "Target account must be a BankAccount" 
            unless blessed($target_account) && $target_account->isa('BankAccount');
        
        $self->withdraw($amount);
        $target_account->deposit($amount);
        
        $self->_add_transaction('transfer_out', $amount, $target_account->account_number);
        $target_account->_add_transaction('transfer_in', $amount, $self->account_number);
        
        return 1;
    }
    
    # Private method for transaction logging
    sub _add_transaction($self, $type, $amount, $details = undef) {
        push @{$self->{transaction_history}}, {
            type => $type,
            amount => $amount,
            timestamp => time(),
            balance_after => $self->{balance},
            details => $details,
        };
    }
    
    # Account summary
    sub summary($self) {
        my $total_deposits = sum(
            map { $_->{amount} } 
            grep { $_->{type} eq 'deposit' } 
            @{$self->{transaction_history}}
        ) || 0;
        
        my $total_withdrawals = sum(
            map { $_->{amount} } 
            grep { $_->{type} eq 'withdrawal' } 
            @{$self->{transaction_history}}
        ) || 0;
        
        return {
            account_number => $self->{account_number},
            current_balance => $self->{balance},
            total_deposits => $total_deposits,
            total_withdrawals => $total_withdrawals,
            transaction_count => scalar(@{$self->{transaction_history}}),
            account_age_days => int((time() - $self->{created_at}) / 86400),
        };
    }
}

# Advanced Data Structures
# ========================

package BinarySearchTree {
    use strict;
    use warnings;
    
    sub new($class) {
        return bless { root => undef }, $class;
    }
    
    sub insert($self, $value) {
        $self->{root} = $self->_insert_node($self->{root}, $value);
    }
    
    sub _insert_node($self, $node, $value) {
        return { value => $value, left => undef, right => undef } unless $node;
        
        if ($value <= $node->{value}) {
            $node->{left} = $self->_insert_node($node->{left}, $value);
        } else {
            $node->{right} = $self->_insert_node($node->{right}, $value);
        }
        
        return $node;
    }
    
    sub search($self, $value) {
        return $self->_search_node($self->{root}, $value);
    }
    
    sub _search_node($self, $node, $value) {
        return 0 unless $node;
        return 1 if $node->{value} == $value;
        
        if ($value < $node->{value}) {
            return $self->_search_node($node->{left}, $value);
        } else {
            return $self->_search_node($node->{right}, $value);
        }
    }
    
    sub inorder_traversal($self) {
        my @result;
        $self->_inorder($self->{root}, \@result);
        return @result;
    }
    
    sub _inorder($self, $node, $result) {
        return unless $node;
        
        $self->_inorder($node->{left}, $result);
        push @$result, $node->{value};
        $self->_inorder($node->{right}, $result);
    }
    
    sub height($self) {
        return $self->_height($self->{root});
    }
    
    sub _height($self, $node) {
        return 0 unless $node;
        
        my $left_height = $self->_height($node->{left});
        my $right_height = $self->_height($node->{right});
        
        return 1 + max($left_height, $right_height);
    }
}

# Graph implementation with adjacency list
package Graph {
    use strict;
    use warnings;
    
    sub new($class) {
        return bless { 
            vertices => {}, 
            edges => {} 
        }, $class;
    }
    
    sub add_vertex($self, $vertex) {
        $self->{vertices}{$vertex} = 1;
        $self->{edges}{$vertex} //= [];
    }
    
    sub add_edge($self, $from, $to, $weight = 1) {
        $self->add_vertex($from);
        $self->add_vertex($to);
        
        push @{$self->{edges}{$from}}, { to => $to, weight => $weight };
    }
    
    sub get_neighbors($self, $vertex) {
        return @{$self->{edges}{$vertex} // []};
    }
    
    sub breadth_first_search($self, $start) {
        my @queue = ($start);
        my %visited = ($start => 1);
        my @result;
        
        while (@queue) {
            my $current = shift @queue;
            push @result, $current;
            
            for my $neighbor ($self->get_neighbors($current)) {
                my $vertex = $neighbor->{to};
                unless ($visited{$vertex}) {
                    $visited{$vertex} = 1;
                    push @queue, $vertex;
                }
            }
        }
        
        return @result;
    }
    
    sub depth_first_search($self, $start) {
        my %visited;
        my @result;
        
        $self->_dfs($start, \%visited, \@result);
        return @result;
    }
    
    sub _dfs($self, $vertex, $visited, $result) {
        $visited->{$vertex} = 1;
        push @$result, $vertex;
        
        for my $neighbor ($self->get_neighbors($vertex)) {
            my $next = $neighbor->{to};
            unless ($visited->{$next}) {
                $self->_dfs($next, $visited, $result);
            }
        }
    }
}

# Functional Programming Utilities
# ================================

# Higher-order functions
sub map_with_index(&@) {
    my $code = shift;
    my @result;
    
    for my $i (0..$#_) {
        push @result, $code->($_[$i], $i);
    }
    
    return @result;
}

sub filter(&@) {
    my $predicate = shift;
    return grep { $predicate->($_) } @_;
}

sub fold(&$@) {
    my ($operation, $initial, @list) = @_;
    my $accumulator = $initial;
    
    for my $item (@list) {
        $accumulator = $operation->($accumulator, $item);
    }
    
    return $accumulator;
}

# Currying implementation
sub curry(&$) {
    my ($sub, $arity) = @_;
    
    return sub {
        my @collected = @_;
        
        if (@collected >= $arity) {
            return $sub->(@collected[0..$arity-1]);
        } else {
            return curry(sub { $sub->(@collected, @_) }, $arity - @collected);
        }
    };
}

# Function composition
sub compose(&@) {
    my @functions = @_;
    
    return sub {
        my $result = $_[0];
        for my $func (reverse @functions) {
            $result = $func->($result);
        }
        return $result;
    };
}

# Memoization
sub memoize(&) {
    my $func = shift;
    my %cache;
    
    return sub {
        my $key = join(',', @_);
        return $cache{$key} //= $func->(@_);
    };
}

# Closures and Advanced References
# ================================

# Counter factory using closures
sub create_counter($initial = 0) {
    my $count = $initial;
    
    return {
        increment => sub { ++$count },
        decrement => sub { --$count },
        get => sub { $count },
        reset => sub { $count = $initial },
    };
}

# Event emitter using closures
sub create_event_emitter() {
    my %listeners;
    
    return {
        on => sub($event, $callback) {
            push @{$listeners{$event}}, $callback;
        },
        
        emit => sub($event, @args) {
            for my $callback (@{$listeners{$event} // []}) {
                $callback->(@args);
            }
        },
        
        off => sub($event, $callback = undef) {
            if (defined $callback) {
                $listeners{$event} = [
                    grep { $_ != $callback } @{$listeners{$event} // []}
                ];
            } else {
                delete $listeners{$event};
            }
        },
    };
}

# File Processing and System Administration
# =========================================

package FileProcessor {
    use strict;
    use warnings;
    use File::Find;
    use File::Basename;
    use Digest::MD5 qw(md5_hex);
    
    # Find duplicate files by MD5 hash
    sub find_duplicates($directory) {
        my %files_by_hash;
        my @duplicates;
        
        find(sub {
            return unless -f $_;
            
            my $hash = _file_md5($_);
            push @{$files_by_hash{$hash}}, $File::Find::name;
        }, $directory);
        
        for my $hash (keys %files_by_hash) {
            if (@{$files_by_hash{$hash}} > 1) {
                push @duplicates, $files_by_hash{$hash};
            }
        }
        
        return @duplicates;
    }
    
    sub _file_md5($filename) {
        open my $fh, '<:raw', $filename or die "Cannot open $filename: $!";
        my $md5 = Digest::MD5->new;
        $md5->addfile($fh);
        close $fh;
        return $md5->hexdigest;
    }
    
    # Analyze file sizes in directory
    sub analyze_directory($directory) {
        my %stats = (
            total_files => 0,
            total_size => 0,
            file_types => {},
            largest_files => [],
        );
        
        find(sub {
            return unless -f $_;
            
            my $size = -s $_;
            my $ext = (fileparse($_, qr/\.[^.]*/))[2] || 'no_extension';
            
            $stats{total_files}++;
            $stats{total_size} += $size;
            $stats{file_types}{$ext}++;
            
            # Keep track of largest files
            push @{$stats{largest_files}}, {
                name => $File::Find::name,
                size => $size,
            };
            
            # Keep only top 10 largest files
            @{$stats{largest_files}} = 
                (sort { $b->{size} <=> $a->{size} } @{$stats{largest_files}})[0..9]
                if @{$stats{largest_files}} > 10;
                
        }, $directory);
        
        return \%stats;
    }
    
    # Process log files with statistics
    sub analyze_log_file($filename) {
        open my $fh, '<', $filename or die "Cannot open $filename: $!";
        
        my %stats = (
            total_lines => 0,
            error_count => 0,
            warning_count => 0,
            ip_addresses => {},
            status_codes => {},
            hourly_distribution => {},
        );
        
        while (my $line = <$fh>) {
            chomp $line;
            $stats{total_lines}++;
            
            # Count error levels
            $stats{error_count}++ if $line =~ /\b(ERROR|error)\b/;
            $stats{warning_count}++ if $line =~ /\b(WARNING|warning)\b/;
            
            # Parse Apache log format
            if (my $parsed = TextProcessor::parse_apache_log($line)) {
                $stats{ip_addresses}{$parsed->{ip}}++;
                $stats{status_codes}{$parsed->{status}}++;
                
                # Extract hour from timestamp
                if ($parsed->{timestamp} =~ /(\d{2}):/) {
                    $stats{hourly_distribution}{$1}++;
                }
            }
        }
        
        close $fh;
        return \%stats;
    }
}

# Performance Optimization and Benchmarking
# ==========================================

package Benchmark {
    use strict;
    use warnings;
    use Time::HiRes qw(time);
    
    sub measure(&$) {
        my ($code, $description) = @_;
        
        my $start = time();
        my $result = $code->();
        my $end = time();
        
        my $duration = $end - $start;
        say sprintf("%-30s: %.6f seconds", $description, $duration);
        
        return $result;
    }
    
    sub compare_implementations($iterations, %implementations) {
        say "\nBenchmarking $iterations iterations:";
        say "-" x 50;
        
        my %results;
        
        for my $name (sort keys %implementations) {
            my $code = $implementations{$name};
            
            my $start = time();
            for (1..$iterations) {
                $code->();
            }
            my $end = time();
            
            my $total_time = $end - $start;
            my $avg_time = $total_time / $iterations;
            
            $results{$name} = {
                total_time => $total_time,
                avg_time => $avg_time,
            };
            
            say sprintf("%-20s: %.6f total, %.9f avg", 
                       $name, $total_time, $avg_time);
        }
        
        return \%results;
    }
}

# Example Usage and Testing
# =========================

sub run_examples() {
    say "=== Advanced Perl Programming Examples ===\n";
    
    # 1. Text Processing
    say "1. Text Processing:";
    my $email = "user@example.com";
    say "   Email '$email' is " . (TextProcessor::validate_email($email) ? "valid" : "invalid");
    
    my $log_line = '192.168.1.1 - - [25/Dec/2023:10:00:00 +0000] "GET /index.html HTTP/1.1" 200 1234';
    my $parsed_log = TextProcessor::parse_apache_log($log_line);
    say "   Parsed log: IP=" . $parsed_log->{ip} . ", Status=" . $parsed_log->{status};
    
    my $text = "The quick brown fox jumps over the lazy dog. The dog was really lazy.";
    my $word_freq = TextProcessor::word_frequency($text);
    say "   Most frequent word: " . $word_freq->[0][0] . " (appears " . $word_freq->[0][1] . " times)";
    
    # 2. Object-Oriented Programming
    say "\n2. Object-Oriented Programming:";
    my $account1 = BankAccount->new(account_number => "12345", initial_balance => 1000);
    my $account2 = BankAccount->new(account_number => "67890", initial_balance => 500);
    
    $account1->deposit(200);
    $account1->withdraw(150);
    $account1->transfer($account2, 300);
    
    my $summary = $account1->summary();
    say sprintf("   Account %s: Balance=%.2f, Transactions=%d", 
               $summary->{account_number}, 
               $summary->{current_balance}, 
               $summary->{transaction_count});
    
    # 3. Data Structures
    say "\n3. Data Structures:";
    
    # Binary Search Tree
    my $bst = BinarySearchTree->new();
    $bst->insert($_) for (50, 30, 70, 20, 40, 60, 80);
    
    my @inorder = $bst->inorder_traversal();
    say "   BST inorder: " . join(", ", @inorder);
    say "   BST height: " . $bst->height();
    say "   BST contains 40: " . ($bst->search(40) ? "yes" : "no");
    
    # Graph
    my $graph = Graph->new();
    $graph->add_edge("A", "B");
    $graph->add_edge("A", "C");
    $graph->add_edge("B", "D");
    $graph->add_edge("C", "E");
    
    my @bfs = $graph->breadth_first_search("A");
    say "   Graph BFS from A: " . join(", ", @bfs);
    
    my @dfs = $graph->depth_first_search("A");
    say "   Graph DFS from A: " . join(", ", @dfs);
    
    # 4. Functional Programming
    say "\n4. Functional Programming:";
    
    my @numbers = (1..10);
    my @squares = map { $_ * $_ } @numbers;
    say "   Squares: " . join(", ", @squares);
    
    my @evens = filter { $_ % 2 == 0 } @numbers;
    say "   Evens: " . join(", ", @evens);
    
    my $sum = fold { $_[0] + $_[1] } 0, @numbers;
    say "   Sum: $sum";
    
    # Memoized fibonacci
    my $fib = memoize(sub {
        my $n = shift;
        return $n if $n <= 1;
        return $fib->($n-1) + $fib->($n-2);
    });
    
    say "   Fibonacci(20): " . $fib->(20);
    
    # 5. Closures
    say "\n5. Closures:";
    
    my $counter = create_counter(10);
    say "   Counter initial: " . $counter->{get}->();
    $counter->{increment}->();
    $counter->{increment}->();
    say "   Counter after 2 increments: " . $counter->{get}->();
    
    my $emitter = create_event_emitter();
    $emitter->{on}->("test", sub { say "   Event triggered with: " . join(", ", @_) });
    $emitter->{emit}->("test", "Hello", "World");
    
    # 6. Performance Comparison
    say "\n6. Performance Comparison:";
    
    Benchmark::compare_implementations(10000,
        "array_push" => sub {
            my @arr;
            push @arr, $_ for 1..100;
        },
        "array_index" => sub {
            my @arr;
            $arr[$_] = $_ for 0..99;
        }
    );
    
    say "\n=== Perl Examples Complete ===";
}

# Advanced Pattern Matching Examples
# ==================================

sub demonstrate_regex_features() {
    say "\n=== Advanced Regex Features ===";
    
    # Recursive regex for balanced parentheses
    my $balanced_parens = qr{
        \(
        (?:
            [^()]++          # Non-parentheses characters
            |
            (?R)             # Recursive call
        )*
        \)
    }x;
    
    my $test_string = "(hello (world) and (nested (deeper) structure))";
    if ($test_string =~ /^$balanced_parens$/) {
        say "   Balanced parentheses: MATCH";
    }
    
    # Named captures with backtracking control
    my $phone_regex = qr{
        (?<country>\+\d{1,3})?      # Optional country code
        \s*
        (?<area>\(\d{3}\)|\d{3})    # Area code with or without parens
        \s*[-.]?\s*
        (?<exchange>\d{3})          # Exchange
        \s*[-.]?\s*
        (?<number>\d{4})            # Number
    }x;
    
    my $phone = "+1 (555) 123-4567";
    if ($phone =~ /$phone_regex/) {
        say "   Phone parsed: Country=$+{country}, Area=$+{area}, Exchange=$+{exchange}, Number=$+{number}";
    }
    
    # Possessive quantifiers for performance
    my $html_tag = qr{<([a-zA-Z][a-zA-Z0-9]*+)(?:\s[^>]*+)?>.*?</\1>}s;
    my $html = "<div class='test'>Content</div>";
    
    if ($html =~ /$html_tag/) {
        say "   HTML tag matched: $1";
    }
}

# Module and Package Development
# =============================

package MyUtils::StringHelper {
    use Exporter 'import';
    our @EXPORT_OK = qw(trim capitalize reverse_words);
    our %EXPORT_TAGS = (
        'all' => [@EXPORT_OK],
        'format' => [qw(trim capitalize)],
    );
    
    sub trim($string) {
        $string =~ s/^\s+|\s+$//g;
        return $string;
    }
    
    sub capitalize($string) {
        return join ' ', map { ucfirst lc } split /\s+/, trim($string);
    }
    
    sub reverse_words($string) {
        return join ' ', reverse split /\s+/, trim($string);
    }
}

# Run examples if script is executed directly
if ($0 eq __FILE__) {
    run_examples();
    demonstrate_regex_features();
}

1;

__END__

=pod

=head1 NAME

Advanced Perl Programming Examples

=head1 DESCRIPTION

This module provides comprehensive examples of advanced Perl programming 
techniques including object-oriented programming, functional programming,
regular expressions, data structures, and performance optimization.

=head1 AUTHOR

Nibert Investments LLC

=head1 LICENSE

This is free software; you can redistribute it and/or modify it under
the same terms as Perl itself.

=cut