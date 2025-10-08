/**
 * Error Handling: Try Catch
 * AI/ML Training Sample
 */

#include <iostream>
#include <string>

class TryCatch {
private:
    std::string data;
    
public:
    TryCatch() : data("") {}
    
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
    TryCatch instance;
    instance.process("example");
    std::cout << "Data: " << instance.getData() << std::endl;
    std::cout << "Valid: " << instance.validate() << std::endl;
    return 0;
}
