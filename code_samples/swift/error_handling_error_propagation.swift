import Foundation

/// Error Handling: Error Propagation
/// AI/ML Training Sample

class ErrorPropagation {
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
let instance = ErrorPropagation()
instance.process("example")
print("Data: \(instance.getData())")
print("Valid: \(instance.validate())")
