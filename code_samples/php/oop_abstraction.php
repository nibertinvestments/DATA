<?php
/**
 * Oop: Abstraction
 * AI/ML Training Sample
 */

class Abstraction {
    private $data;
    
    public function __construct() {
        $this->data = "";
    }
    
    public function process($input) {
        $this->data = $input;
    }
    
    public function getData() {
        return $this->data;
    }
    
    public function validate() {
        return !empty($this->data);
    }
}

// Example usage
$instance = new Abstraction();
$instance->process("example");
echo "Data: " . $instance->getData() . "\n";
echo "Valid: " . ($instance->validate() ? "true" : "false") . "\n";
?>
