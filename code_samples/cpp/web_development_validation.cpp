/**
 * Web Development: Validation
 * AI/ML Training Sample
 */

#include <iostream>
#include <string>

class Validation {
private:
    std::string data;
    
public:
    Validation() : data("") {}
    
    void process(const std::string& input) {
        data = input;
    }
    
    std::string getData() const {
        return data;
    }
    
    bool validate() const {
        return !data.empty();
    }
};

int main() {
    Validation instance;
    instance.process("example");
    std::cout << "Data: " << instance.getData() << std::endl;
    std::cout << "Valid: " << instance.validate() << std::endl;
    return 0;
}
