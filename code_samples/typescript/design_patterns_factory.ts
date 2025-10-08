/**
 * Design Patterns: Factory
 * AI/ML Training Sample
 */

interface IFactory {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class Factory implements IFactory {
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
const instance = new Factory();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { Factory, IFactory };
