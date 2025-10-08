import Foundation

/// Data Structures: Linked List
/// AI/ML Training Sample

class LinkedList {
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
let instance = LinkedList()
instance.process("example")
print("Data: \(instance.getData())")
print("Valid: \(instance.validate())")
