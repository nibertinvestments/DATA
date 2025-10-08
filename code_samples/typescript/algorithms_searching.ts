/**
 * Algorithms: Searching
 * AI/ML Training Sample
 */

interface ISearching {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class Searching implements ISearching {
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
const instance = new Searching();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { Searching, ISearching };
