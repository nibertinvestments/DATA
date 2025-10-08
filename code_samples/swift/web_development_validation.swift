import Foundation

/// Web Development: Validation
/// AI/ML Training Sample

class Validation {
    var data: String
    
    init() {
        self.data = ""
    }
    
    func process(_ input: String) {
        self.data = input
    }
    
    func getData() -> String {
        return self.data
    }
    
    func validate() -> Bool {
        return !self.data.isEmpty
    }
}

// Example usage
let instance = Validation()
instance.process("example")
print("Data: \(instance.getData())")
print("Valid: \(instance.validate())")
