/**
 * Networking: Http Client
 * AI/ML Training Sample
 */

interface IHttpClient {
    data: string;
    process(input: string): void;
    validate(): boolean;
}

class HttpClient implements IHttpClient {
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
const instance = new HttpClient();
instance.process("example");
console.log(`Data: ${instance.getData()}`);
console.log(`Valid: ${instance.validate()}`);

export { HttpClient, IHttpClient };
