/**
 * Error Handling: Custom Exceptions
 * AI/ML Training Sample
 */

interface ICustomExceptions {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class CustomExceptions implements ICustomExceptions {
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
const instance = new CustomExceptions();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { CustomExceptions, ICustomExceptions };
