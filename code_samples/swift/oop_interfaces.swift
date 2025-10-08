import Foundation

/// Oop: Interfaces
/// AI/ML Training Sample

class Interfaces {
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
let instance = Interfaces()
instance.process("example")
print("Data: \(instance.getData())")
print("Valid: \(instance.validate())")
