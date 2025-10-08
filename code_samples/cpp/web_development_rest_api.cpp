/**
 * Web Development: Rest Api
 * AI/ML Training Sample
 */

#include <iostream>
#include <string>

class RestApi {
private:
    std::string data;
    
public:
    RestApi() : data("") {}
    
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
    RestApi instance;
    instance.process("example");
    std::cout << "Data: " << instance.getData() << std::endl;
    std::cout << "Valid: " << instance.validate() << std::endl;
    return 0;
}
