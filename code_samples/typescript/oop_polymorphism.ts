/**
 * Oop: Polymorphism
 * AI/ML Training Sample
 */

interface IPolymorphism {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class Polymorphism implements IPolymorphism {
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
const instance = new Polymorphism();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { Polymorphism, IPolymorphism };
