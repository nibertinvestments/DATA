/**
 * Design Patterns: Decorator
 * AI/ML Training Sample
 */

interface IDecorator {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class Decorator implements IDecorator {
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
const instance = new Decorator();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { Decorator, IDecorator };
