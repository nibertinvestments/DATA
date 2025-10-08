//! Algorithms: Sorting
//! AI/ML Training Sample

pub struct Sorting {
    data: String,
}

impl Sorting {
    pub fn new() -> Self {
        Self {
            data: String::new(),
        }
    }
    
    pub fn process(&mut self, input: &str) {
        self.data = input.to_string();
    }
    
    pub fn get_data(&self) -> &str {
        &self.data
    }
    
    pub fn validate(&self) -> bool {
        !self.data.is_empty()
    }
}

fn main() {
    let mut instance = Sorting::new();
    instance.process("example");
    println!("Data: {}", instance.get_data());
    println!("Valid: {}", instance.validate());
}
