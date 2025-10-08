/**
 * Performance: Lazy Loading
 * AI/ML Training Sample
 */

#include <iostream>
#include <string>

class LazyLoading {
private:
    std::string data;
    
public:
    LazyLoading() : data("") {}
    
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
    LazyLoading instance;
    instance.process("example");
    std::cout << "Data: " << instance.getData() << std::endl;
    std::cout << "Valid: " << instance.validate() << std::endl;
    return 0;
}
