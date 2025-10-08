/**
 * Oop: Interfaces
 * AI/ML Training Sample
 */

#include <iostream>
#include <string>

class Interfaces {
private:
    std::string data;
    
public:
    Interfaces() : data("") {}
    
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
    Interfaces instance;
    instance.process("example");
    std::cout << "Data: " << instance.getData() << std::endl;
    std::cout << "Valid: " << instance.validate() << std::endl;
    return 0;
}
