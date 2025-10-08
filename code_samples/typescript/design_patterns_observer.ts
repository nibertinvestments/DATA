/**
 * Design Patterns: Observer
 * AI/ML Training Sample
 */

interface IObserver {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class Observer implements IObserver {
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
const instance = new Observer();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { Observer, IObserver };
