/**
 * Functional: Monads
 * AI/ML Training Sample
 */

#include <iostream>
#include <string>

class Monads {
private:
    std::string data;
    
public:
    Monads() : data("") {}
    
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
    Monads instance;
    instance.process("example");
    std::cout << "Data: " << instance.getData() << std::endl;
    std::cout << "Valid: " << instance.validate() << std::endl;
    return 0;
}
