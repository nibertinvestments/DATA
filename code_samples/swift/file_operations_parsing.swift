import Foundation

/// File Operations: Parsing
/// AI/ML Training Sample

class Parsing {
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
let instance = Parsing()
instance.process("example")
print("Data: \(instance.getData())")
print("Valid: \(instance.validate())")
