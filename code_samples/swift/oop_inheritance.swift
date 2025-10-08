import Foundation

/// Oop: Inheritance
/// AI/ML Training Sample

class Inheritance {
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
let instance = Inheritance()
instance.process("example")
print("Data: \(instance.getData())")
print("Valid: \(instance.validate())")
