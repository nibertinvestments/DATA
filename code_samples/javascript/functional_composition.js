/**
 * Functional Programming Composition Patterns
 * Demonstrates pure functions, composition, and functional utilities
 */

// Pure function utilities
const identity = x => x;
const constant = x => () => x;
const compose = (...fns) => x => fns.reduceRight((v, f) => f(v), x);
const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);

// Currying
const curry = (fn) => {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        } else {
            return function(...args2) {
                return curried.apply(this, args.concat(args2));
            };
        }
    };
};

// Partial application
const partial = (fn, ...presetArgs) => {
    return (...laterArgs) => fn(...presetArgs, ...laterArgs);
};

// Function composition utilities
const map = curry((fn, arr) => arr.map(fn));
const filter = curry((fn, arr) => arr.filter(fn));
const reduce = curry((fn, init, arr) => arr.reduce(fn, init));

// Transducers
const mapping = (fn) => (step) => {
    return (acc, value) => step(acc, fn(value));
};

const filtering = (predicate) => (step) => {
    return (acc, value) => predicate(value) ? step(acc, value) : acc;
};

const transduce = (xform, step, init, coll) => {
    const xf = xform(step);
    return coll.reduce(xf, init);
};

// Monads
class Maybe {
    constructor(value) {
        this.value = value;
    }
    
    static of(value) {
        return new Maybe(value);
    }
    
    isNothing() {
        return this.value === null || this.value === undefined;
    }
    
    map(fn) {
        return this.isNothing() ? this : Maybe.of(fn(this.value));
    }
    
    flatMap(fn) {
        return this.isNothing() ? this : fn(this.value);
    }
    
    getOrElse(defaultValue) {
        return this.isNothing() ? defaultValue : this.value;
    }
}

class Either {
    constructor(value, isRight = true) {
        this.value = value;
        this.isRight = isRight;
    }
    
    static right(value) {
        return new Either(value, true);
    }
    
    static left(value) {
        return new Either(value, false);
    }
    
    map(fn) {
        return this.isRight ? Either.right(fn(this.value)) : this;
    }
    
    flatMap(fn) {
        return this.isRight ? fn(this.value) : this;
    }
    
    fold(leftFn, rightFn) {
        return this.isRight ? rightFn(this.value) : leftFn(this.value);
    }
}

// Lens implementation
const lens = (getter, setter) => ({
    get: getter,
    set: setter,
    over: (fn, obj) => setter(fn(getter(obj)), obj)
});

const view = (lens, obj) => lens.get(obj);
const set = (lens, value, obj) => lens.set(value, obj);
const over = (lens, fn, obj) => lens.over(fn, obj);

// Demonstration
function main() {
    console.log("Functional Composition Patterns");
    console.log("===============================\n");
    
    // Composition example
    const double = x => x * 2;
    const increment = x => x + 1;
    const square = x => x * x;
    
    const transform = pipe(double, increment, square);
    console.log("1. Function composition:");
    console.log(`   pipe(double, increment, square)(3) = ${transform(3)}\n`);
    
    // Currying example
    const add = curry((a, b, c) => a + b + c);
    console.log("2. Currying:");
    console.log(`   add(1)(2)(3) = ${add(1)(2)(3)}\n`);
    
    // Maybe monad
    console.log("3. Maybe monad:");
    const safeDiv = (a, b) => b === 0 ? Maybe.of(null) : Maybe.of(a / b);
    console.log(`   10 / 2 = ${safeDiv(10, 2).getOrElse("Error")}`);
    console.log(`   10 / 0 = ${safeDiv(10, 0).getOrElse("Error")}\n`);
    
    // Transducers
    console.log("4. Transducers:");
    const xform = compose(
        filtering(x => x % 2 === 0),
        mapping(x => x * 2)
    );
    const result = transduce(
        xform,
        (acc, x) => acc.concat(x),
        [],
        [1, 2, 3, 4, 5, 6]
    );
    console.log(`   Result: [${result}]`);
}

main();
