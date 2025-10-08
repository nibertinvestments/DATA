<?php
/**
 * Networking: Protocols
 * AI/ML Training Sample
 */

class Protocols {
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
$instance = new Protocols();
$instance->process("example");
echo "Data: " . $instance->getData() . "\n";
echo "Valid: " . ($instance->validate() ? "true" : "false") . "\n";
?>
