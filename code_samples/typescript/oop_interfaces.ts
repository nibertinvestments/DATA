/**
 * Oop: Interfaces
 * AI/ML Training Sample
 */

interface IInterfaces {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class Interfaces implements IInterfaces {
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
const instance = new Interfaces();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { Interfaces, IInterfaces };
