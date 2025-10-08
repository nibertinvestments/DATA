/**
 * Data Structures: Heap
 * AI/ML Training Sample
 */

interface IHeap {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class Heap implements IHeap {
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
const instance = new Heap();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { Heap, IHeap };
