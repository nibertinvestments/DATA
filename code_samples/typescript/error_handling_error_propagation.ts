/**
 * Error Handling: Error Propagation
 * AI/ML Training Sample
 */

interface IErrorPropagation {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class ErrorPropagation implements IErrorPropagation {
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
const instance = new ErrorPropagation();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { ErrorPropagation, IErrorPropagation };
