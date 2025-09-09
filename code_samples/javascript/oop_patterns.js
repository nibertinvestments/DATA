// Object-oriented programming patterns in JavaScript

console.log("=== JavaScript OOP Patterns ===");

// Classical inheritance with prototypes
function Animal(name, species) {
    this.name = name;
    this.species = species;
    this.energy = 100;
}

Animal.prototype.eat = function(food) {
    console.log(`${this.name} eats ${food}`);
    this.energy += 10;
    return this;
};

Animal.prototype.sleep = function(hours) {
    console.log(`${this.name} sleeps for ${hours} hours`);
    this.energy += hours * 5;
    return this;
};

Animal.prototype.move = function(distance) {
    console.log(`${this.name} moves ${distance} meters`);
    this.energy -= distance * 0.1;
    return this;
};

// Inheritance with prototypes
function Dog(name, breed) {
    Animal.call(this, name, 'Canine');
    this.breed = breed;
}

Dog.prototype = Object.create(Animal.prototype);
Dog.prototype.constructor = Dog;

Dog.prototype.bark = function() {
    console.log(`${this.name} barks: Woof!`);
    this.energy -= 2;
    return this;
};

Dog.prototype.fetch = function(item) {
    console.log(`${this.name} fetches the ${item}`);
    this.energy -= 5;
    return this;
};

// Factory pattern
const AnimalFactory = {
    createAnimal(type, name, ...args) {
        switch(type.toLowerCase()) {
            case 'dog':
                return new Dog(name, args[0]);
            case 'cat':
                return new Cat(name, args[0]);
            case 'bird':
                return new Bird(name, args[0]);
            default:
                return new Animal(name, type);
        }
    },
    
    createMultiple(specifications) {
        return specifications.map(spec => 
            this.createAnimal(spec.type, spec.name, ...spec.args)
        );
    }
};

// Mixin pattern
const Flyable = {
    fly(distance) {
        console.log(`${this.name} flies ${distance} meters`);
        this.energy -= distance * 0.05;
        this.altitude = (this.altitude || 0) + distance;
        return this;
    },
    
    land() {
        console.log(`${this.name} lands`);
        this.altitude = 0;
        return this;
    }
};

const Swimmable = {
    swim(distance) {
        console.log(`${this.name} swims ${distance} meters`);
        this.energy -= distance * 0.03;
        return this;
    },
    
    dive(depth) {
        console.log(`${this.name} dives ${depth} meters deep`);
        this.energy -= depth * 0.2;
        return this;
    }
};

function Bird(name, species) {
    Animal.call(this, name, 'Avian');
    this.species = species;
    this.altitude = 0;
}

Bird.prototype = Object.create(Animal.prototype);
Bird.prototype.constructor = Bird;

// Apply mixin to Bird
Object.assign(Bird.prototype, Flyable);

Bird.prototype.chirp = function() {
    console.log(`${this.name} chirps: Tweet tweet!`);
    this.energy -= 1;
    return this;
};

// Module pattern with revealing module
const ZooManager = (function() {
    // Private variables
    let animals = [];
    let capacity = 50;
    let nextId = 1;
    
    // Private methods
    function findAnimalById(id) {
        return animals.find(animal => animal.id === id);
    }
    
    function validateCapacity() {
        return animals.length < capacity;
    }
    
    function generateReport() {
        const report = {
            totalAnimals: animals.length,
            capacity: capacity,
            remaining: capacity - animals.length,
            species: {}
        };
        
        animals.forEach(animal => {
            const species = animal.species;
            report.species[species] = (report.species[species] || 0) + 1;
        });
        
        return report;
    }
    
    // Public API
    return {
        addAnimal(animal) {
            if (!validateCapacity()) {
                throw new Error('Zoo is at full capacity');
            }
            
            animal.id = nextId++;
            animal.addedDate = new Date();
            animals.push(animal);
            
            console.log(`Added ${animal.name} to the zoo`);
            return animal.id;
        },
        
        removeAnimal(id) {
            const index = animals.findIndex(animal => animal.id === id);
            if (index === -1) {
                throw new Error('Animal not found');
            }
            
            const removed = animals.splice(index, 1)[0];
            console.log(`Removed ${removed.name} from the zoo`);
            return removed;
        },
        
        getAnimal(id) {
            return findAnimalById(id);
        },
        
        getAllAnimals() {
            return [...animals]; // Return copy to prevent external modification
        },
        
        feedAllAnimals(food) {
            animals.forEach(animal => {
                if (typeof animal.eat === 'function') {
                    animal.eat(food);
                }
            });
        },
        
        getReport() {
            return generateReport();
        },
        
        setCapacity(newCapacity) {
            if (newCapacity < animals.length) {
                throw new Error('Cannot set capacity below current animal count');
            }
            capacity = newCapacity;
        },
        
        // Method chaining for operations
        performAction(animalId, action, ...args) {
            const animal = findAnimalById(animalId);
            if (!animal) {
                throw new Error('Animal not found');
            }
            
            if (typeof animal[action] === 'function') {
                animal[action](...args);
            } else {
                throw new Error(`Action ${action} not available for this animal`);
            }
            
            return this; // Enable chaining
        }
    };
})();

// Observer pattern
class Subject {
    constructor() {
        this.observers = [];
    }
    
    subscribe(observer) {
        this.observers.push(observer);
    }
    
    unsubscribe(observer) {
        this.observers = this.observers.filter(obs => obs !== observer);
    }
    
    notify(data) {
        this.observers.forEach(observer => observer.update(data));
    }
}

class AnimalHealthMonitor extends Subject {
    constructor() {
        super();
        this.animalData = new Map();
    }
    
    monitorAnimal(animal) {
        const healthData = {
            id: animal.id,
            name: animal.name,
            energy: animal.energy,
            lastChecked: new Date(),
            status: this.getHealthStatus(animal.energy)
        };
        
        this.animalData.set(animal.id, healthData);
        this.notify(healthData);
    }
    
    getHealthStatus(energy) {
        if (energy > 80) return 'Excellent';
        if (energy > 60) return 'Good';
        if (energy > 40) return 'Fair';
        if (energy > 20) return 'Poor';
        return 'Critical';
    }
    
    getAnimalHealth(animalId) {
        return this.animalData.get(animalId);
    }
}

// Observer implementations
class VeterinarianObserver {
    constructor(name) {
        this.name = name;
    }
    
    update(healthData) {
        if (healthData.status === 'Poor' || healthData.status === 'Critical') {
            console.log(`🏥 Dr. ${this.name}: ${healthData.name} needs immediate attention! Energy: ${healthData.energy}`);
        }
    }
}

class ZookeeperObserver {
    constructor(name) {
        this.name = name;
        this.careLog = [];
    }
    
    update(healthData) {
        this.careLog.push({
            ...healthData,
            caretaker: this.name
        });
        
        if (healthData.energy < 50) {
            console.log(`🧑‍🦲 ${this.name}: ${healthData.name} might need feeding. Energy: ${healthData.energy}`);
        }
    }
    
    getCareLog() {
        return this.careLog;
    }
}

// Strategy pattern
class FeedingStrategy {
    feed(animal, amount) {
        throw new Error('Feed method must be implemented');
    }
}

class CarnivoreFeeding extends FeedingStrategy {
    feed(animal, amount) {
        console.log(`🥩 Feeding ${animal.name} ${amount}kg of meat`);
        animal.energy += amount * 15;
    }
}

class HerbivoreFeeding extends FeedingStrategy {
    feed(animal, amount) {
        console.log(`🥬 Feeding ${animal.name} ${amount}kg of vegetables`);
        animal.energy += amount * 10;
    }
}

class OmnivoreFeeding extends FeedingStrategy {
    feed(animal, amount) {
        const meatAmount = amount * 0.6;
        const vegAmount = amount * 0.4;
        console.log(`🍽️ Feeding ${animal.name} ${meatAmount}kg meat and ${vegAmount}kg vegetables`);
        animal.energy += amount * 12;
    }
}

class FeedingContext {
    constructor(strategy) {
        this.strategy = strategy;
    }
    
    setStrategy(strategy) {
        this.strategy = strategy;
    }
    
    feedAnimal(animal, amount) {
        this.strategy.feed(animal, amount);
    }
}

// Command pattern
class Command {
    execute() {
        throw new Error('Execute method must be implemented');
    }
    
    undo() {
        throw new Error('Undo method must be implemented');
    }
}

class FeedAnimalCommand extends Command {
    constructor(animal, food, amount) {
        super();
        this.animal = animal;
        this.food = food;
        this.amount = amount;
        this.previousEnergy = animal.energy;
    }
    
    execute() {
        console.log(`Executing: Feed ${this.animal.name} ${this.amount} ${this.food}`);
        this.animal.eat(`${this.amount} ${this.food}`);
    }
    
    undo() {
        console.log(`Undoing: Feed ${this.animal.name}`);
        this.animal.energy = this.previousEnergy;
    }
}

class MoveAnimalCommand extends Command {
    constructor(animal, distance) {
        super();
        this.animal = animal;
        this.distance = distance;
        this.previousEnergy = animal.energy;
    }
    
    execute() {
        console.log(`Executing: Move ${this.animal.name} ${this.distance}m`);
        this.animal.move(this.distance);
    }
    
    undo() {
        console.log(`Undoing: Move ${this.animal.name}`);
        this.animal.energy = this.previousEnergy;
    }
}

class CommandInvoker {
    constructor() {
        this.history = [];
        this.currentPosition = -1;
    }
    
    execute(command) {
        // Remove any commands after current position
        this.history = this.history.slice(0, this.currentPosition + 1);
        
        command.execute();
        this.history.push(command);
        this.currentPosition++;
    }
    
    undo() {
        if (this.currentPosition >= 0) {
            const command = this.history[this.currentPosition];
            command.undo();
            this.currentPosition--;
        }
    }
    
    redo() {
        if (this.currentPosition < this.history.length - 1) {
            this.currentPosition++;
            const command = this.history[this.currentPosition];
            command.execute();
        }
    }
    
    getHistory() {
        return this.history.map(cmd => cmd.constructor.name);
    }
}

// Decorator pattern
class AnimalDecorator {
    constructor(animal) {
        this.animal = animal;
    }
    
    // Delegate all calls to the wrapped animal
    get name() { return this.animal.name; }
    get species() { return this.animal.species; }
    get energy() { return this.animal.energy; }
    set energy(value) { this.animal.energy = value; }
    
    eat(food) { return this.animal.eat(food); }
    sleep(hours) { return this.animal.sleep(hours); }
    move(distance) { return this.animal.move(distance); }
}

class TrainedAnimalDecorator extends AnimalDecorator {
    constructor(animal, tricks = []) {
        super(animal);
        this.tricks = tricks;
        this.trainingLevel = 1;
    }
    
    performTrick(trick) {
        if (this.tricks.includes(trick)) {
            console.log(`🎪 ${this.name} performs ${trick}!`);
            this.animal.energy -= 3;
            return true;
        } else {
            console.log(`${this.name} doesn't know how to ${trick}`);
            return false;
        }
    }
    
    learnTrick(trick) {
        if (!this.tricks.includes(trick)) {
            this.tricks.push(trick);
            console.log(`${this.name} learned ${trick}!`);
            this.trainingLevel++;
        }
    }
    
    getTricks() {
        return [...this.tricks];
    }
}

class MedicalAnimalDecorator extends AnimalDecorator {
    constructor(animal) {
        super(animal);
        this.medications = [];
        this.vaccinations = [];
        this.healthRecords = [];
    }
    
    administer(medication, dosage) {
        const record = {
            type: 'medication',
            medication,
            dosage,
            date: new Date(),
            administeredBy: 'Dr. Smith'
        };
        
        this.medications.push(record);
        this.healthRecords.push(record);
        console.log(`💊 Administered ${dosage} of ${medication} to ${this.name}`);
    }
    
    vaccinate(vaccine) {
        const record = {
            type: 'vaccination',
            vaccine,
            date: new Date(),
            administeredBy: 'Dr. Johnson'
        };
        
        this.vaccinations.push(record);
        this.healthRecords.push(record);
        console.log(`💉 Vaccinated ${this.name} with ${vaccine}`);
    }
    
    getHealthHistory() {
        return [...this.healthRecords];
    }
}

// Demonstration and testing
function demonstrateOOPPatterns() {
    console.log("\n=== Creating Animals ===");
    
    // Create animals using factory
    const animals = AnimalFactory.createMultiple([
        { type: 'dog', name: 'Buddy', args: ['Golden Retriever'] },
        { type: 'bird', name: 'Tweety', args: ['Canary'] },
        { type: 'dog', name: 'Rex', args: ['German Shepherd'] }
    ]);
    
    console.log("\n=== Zoo Management ===");
    
    // Add animals to zoo
    animals.forEach(animal => ZooManager.addAnimal(animal));
    
    // Display zoo report
    console.log('Zoo Report:', ZooManager.getReport());
    
    console.log("\n=== Health Monitoring ===");
    
    // Set up health monitoring
    const healthMonitor = new AnimalHealthMonitor();
    const vet = new VeterinarianObserver('Wilson');
    const zookeeper = new ZookeeperObserver('Alice');
    
    healthMonitor.subscribe(vet);
    healthMonitor.subscribe(zookeeper);
    
    // Monitor animal health
    animals.forEach(animal => {
        animal.energy = Math.floor(Math.random() * 100); // Random energy for demo
        healthMonitor.monitorAnimal(animal);
    });
    
    console.log("\n=== Feeding Strategies ===");
    
    // Feeding with strategy pattern
    const carnivoreFeeding = new CarnivoreFeeding();
    const herbivoreFeeding = new HerbivoreFeeding();
    const feedingContext = new FeedingContext(carnivoreFeeding);
    
    // Feed the dog (carnivore)
    feedingContext.feedAnimal(animals[0], 2);
    
    // Switch strategy for herbivore
    feedingContext.setStrategy(herbivoreFeeding);
    feedingContext.feedAnimal(animals[1], 1.5);
    
    console.log("\n=== Command Pattern ===");
    
    // Command pattern demonstration
    const invoker = new CommandInvoker();
    
    const feedCommand = new FeedAnimalCommand(animals[0], 'treats', 1);
    const moveCommand = new MoveAnimalCommand(animals[0], 10);
    
    invoker.execute(feedCommand);
    invoker.execute(moveCommand);
    
    console.log('Command history:', invoker.getHistory());
    
    // Undo operations
    invoker.undo();
    invoker.undo();
    
    console.log("\n=== Decorator Pattern ===");
    
    // Decorate animals with additional capabilities
    const trainedBuddy = new TrainedAnimalDecorator(animals[0], ['sit', 'roll']);
    trainedBuddy.learnTrick('fetch');
    trainedBuddy.performTrick('sit');
    trainedBuddy.performTrick('roll');
    console.log('Tricks:', trainedBuddy.getTricks());
    
    const medicalRex = new MedicalAnimalDecorator(animals[2]);
    medicalRex.administer('Antibiotics', '500mg');
    medicalRex.vaccinate('Rabies');
    console.log('Health history:', medicalRex.getHealthHistory());
    
    console.log("\n=== Method Chaining ===");
    
    // Demonstrate method chaining
    animals[0]
        .eat('kibble')
        .move(5)
        .sleep(2)
        .move(3);
    
    console.log(`Final energy: ${animals[0].energy}`);
    
    console.log('\nOOP Patterns demonstration completed!');
}

// Run the demonstration
demonstrateOOPPatterns();