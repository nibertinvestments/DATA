/**
 * Functional: Higher Order
 * AI/ML Training Sample
 */

interface IHigherOrder {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class HigherOrder implements IHigherOrder {
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
const instance = new HigherOrder();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { HigherOrder, IHigherOrder };
