/**
 * Design Patterns: Adapter
 * AI/ML Training Sample
 */

interface IAdapter {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class Adapter implements IAdapter {
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
const instance = new Adapter();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { Adapter, IAdapter };
