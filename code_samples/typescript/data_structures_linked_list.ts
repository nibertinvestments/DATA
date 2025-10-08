/**
 * Data Structures: Linked List
 * AI/ML Training Sample
 */

interface ILinkedList {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class LinkedList implements ILinkedList {
    data: string;
    
    constructor() {
        this.data = "";
    }
    
    process(input: string): void {
        this.data = input;
    }
    
    getData(): string {
        return this.data;
    }
    
    validate(): boolean {
        return this.data.length > 0;
    }
}

// Example usage
const instance = new LinkedList();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { LinkedList, ILinkedList };
