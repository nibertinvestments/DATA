/**
 * Security: Input Validation
 * AI/ML Training Sample
 */

interface IInputValidation {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class InputValidation implements IInputValidation {
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
const instance = new InputValidation();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { InputValidation, IInputValidation };
