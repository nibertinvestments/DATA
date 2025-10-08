/**
 * Oop: Abstraction
 * AI/ML Training Sample
 */

#include <iostream>
#include <string>

class Abstraction {
private:
    std::string data;
    
public:
    Abstraction() : data("") {}
    
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
    Abstraction instance;
    instance.process("example");
    std::cout << "Data: " << instance.getData() << std::endl;
    std::cout << "Valid: " << instance.validate() << std::endl;
    return 0;
}
