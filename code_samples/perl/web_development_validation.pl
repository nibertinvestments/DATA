#!/usr/bin/perl
# Web Development: Validation
# AI/ML Training Sample

package Validation;
use strict;
use warnings;

sub new {
    my $class = shift;
    my $self = {
        data => '',
    };
    bless $self, $class;
    return $self;
}

sub process {
    my ($self, $input) = @_;
    $self->{data} = $input;
}

sub get_data {
    my $self = shift;
    return $self->{data};
}

sub validate {
    my $self = shift;
    return length($self->{data}) > 0;
}

# Example usage
my $instance = Validation->new();
$instance->process("example");
print "Data: " . $instance->get_data() . "\n";
print "Valid: " . ($instance->validate() ? "true" : "false") . "\n";

1;
