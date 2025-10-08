/**
 * Functional: Currying
 * AI/ML Training Sample
 */

interface ICurrying {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class Currying implements ICurrying {
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
const instance = new Currying();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { Currying, ICurrying };
