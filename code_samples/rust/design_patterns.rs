// Rust design patterns implementation for AI training dataset.
// Demonstrates common design patterns in Rust.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Singleton pattern using lazy_static and Mutex.
use std::sync::Once;

static INIT: Once = Once::new();
static mut SINGLETON: Option<Box<DatabaseConnection>> = None;

pub struct DatabaseConnection {
    connection_string: String,
}

impl DatabaseConnection {
    fn new(connection_string: String) -> Self {
        DatabaseConnection { connection_string }
    }

    pub fn get_instance() -> &'static DatabaseConnection {
        unsafe {
            INIT.call_once(|| {
                SINGLETON = Some(Box::new(DatabaseConnection::new(
                    "database://localhost:5432".to_string(),
                )));
            });
            SINGLETON.as_ref().unwrap()
        }
    }

    pub fn execute_query(&self, query: &str) -> String {
        format!("Executing '{}' on {}", query, self.connection_string)
    }
}

/// Builder pattern for constructing complex objects.
#[derive(Debug, Clone)]
pub struct Computer {
    cpu: String,
    ram: u32,
    storage: u32,
    gpu: Option<String>,
    bluetooth: bool,
    wifi: bool,
}

pub struct ComputerBuilder {
    cpu: Option<String>,
    ram: Option<u32>,
    storage: Option<u32>,
    gpu: Option<String>,
    bluetooth: bool,
    wifi: bool,
}

impl ComputerBuilder {
    pub fn new() -> Self {
        ComputerBuilder {
            cpu: None,
            ram: None,
            storage: None,
            gpu: None,
            bluetooth: false,
            wifi: false,
        }
    }

    pub fn cpu(mut self, cpu: String) -> Self {
        self.cpu = Some(cpu);
        self
    }

    pub fn ram(mut self, ram: u32) -> Self {
        self.ram = Some(ram);
        self
    }

    pub fn storage(mut self, storage: u32) -> Self {
        self.storage = Some(storage);
        self
    }

    pub fn gpu(mut self, gpu: String) -> Self {
        self.gpu = Some(gpu);
        self
    }

    pub fn bluetooth(mut self, bluetooth: bool) -> Self {
        self.bluetooth = bluetooth;
        self
    }

    pub fn wifi(mut self, wifi: bool) -> Self {
        self.wifi = wifi;
        self
    }

    pub fn build(self) -> Result<Computer, String> {
        let cpu = self.cpu.ok_or("CPU is required")?;
        let ram = self.ram.ok_or("RAM is required")?;
        let storage = self.storage.ok_or("Storage is required")?;

        Ok(Computer {
            cpu,
            ram,
            storage,
            gpu: self.gpu,
            bluetooth: self.bluetooth,
            wifi: self.wifi,
        })
    }
}

/// Factory pattern for creating different types of vehicles.
pub trait Vehicle {
    fn start(&self) -> String;
    fn stop(&self) -> String;
    fn get_type(&self) -> String;
}

pub struct Car {
    brand: String,
}

impl Car {
    pub fn new(brand: String) -> Self {
        Car { brand }
    }
}

impl Vehicle for Car {
    fn start(&self) -> String {
        format!("Starting {} car engine", self.brand)
    }

    fn stop(&self) -> String {
        format!("Stopping {} car engine", self.brand)
    }

    fn get_type(&self) -> String {
        "Car".to_string()
    }
}

pub struct Motorcycle {
    brand: String,
}

impl Motorcycle {
    pub fn new(brand: String) -> Self {
        Motorcycle { brand }
    }
}

impl Vehicle for Motorcycle {
    fn start(&self) -> String {
        format!("Starting {} motorcycle engine", self.brand)
    }

    fn stop(&self) -> String {
        format!("Stopping {} motorcycle engine", self.brand)
    }

    fn get_type(&self) -> String {
        "Motorcycle".to_string()
    }
}

pub struct VehicleFactory;

impl VehicleFactory {
    pub fn create_vehicle(vehicle_type: &str, brand: String) -> Option<Box<dyn Vehicle>> {
        match vehicle_type {
            "car" => Some(Box::new(Car::new(brand))),
            "motorcycle" => Some(Box::new(Motorcycle::new(brand))),
            _ => None,
        }
    }
}

/// Observer pattern for event notification.
pub trait Observer {
    fn update(&self, message: &str);
}

pub struct EmailNotifier {
    email: String,
}

impl EmailNotifier {
    pub fn new(email: String) -> Self {
        EmailNotifier { email }
    }
}

impl Observer for EmailNotifier {
    fn update(&self, message: &str) {
        println!("Email notification to {}: {}", self.email, message);
    }
}

pub struct SMSNotifier {
    phone: String,
}

impl SMSNotifier {
    pub fn new(phone: String) -> Self {
        SMSNotifier { phone }
    }
}

impl Observer for SMSNotifier {
    fn update(&self, message: &str) {
        println!("SMS notification to {}: {}", self.phone, message);
    }
}

pub struct Subject {
    observers: Vec<Box<dyn Observer>>,
    state: String,
}

impl Subject {
    pub fn new() -> Self {
        Subject {
            observers: Vec::new(),
            state: String::new(),
        }
    }

    pub fn attach(&mut self, observer: Box<dyn Observer>) {
        self.observers.push(observer);
    }

    pub fn detach(&mut self, index: usize) {
        if index < self.observers.len() {
            self.observers.remove(index);
        }
    }

    pub fn notify(&self) {
        for observer in &self.observers {
            observer.update(&self.state);
        }
    }

    pub fn set_state(&mut self, state: String) {
        self.state = state;
        self.notify();
    }

    pub fn get_state(&self) -> &str {
        &self.state
    }
}

/// Strategy pattern for different algorithms.
pub trait SortStrategy {
    fn sort(&self, data: &mut Vec<i32>);
}

pub struct BubbleSort;

impl SortStrategy for BubbleSort {
    fn sort(&self, data: &mut Vec<i32>) {
        let n = data.len();
        for i in 0..n {
            for j in 0..n - i - 1 {
                if data[j] > data[j + 1] {
                    data.swap(j, j + 1);
                }
            }
        }
    }
}

pub struct QuickSort;

impl SortStrategy for QuickSort {
    fn sort(&self, data: &mut Vec<i32>) {
        if data.len() <= 1 {
            return;
        }
        quick_sort_recursive(data, 0, data.len() - 1);
    }
}

fn quick_sort_recursive(arr: &mut Vec<i32>, low: usize, high: usize) {
    if low < high {
        let pi = partition(arr, low, high);
        if pi > 0 {
            quick_sort_recursive(arr, low, pi - 1);
        }
        quick_sort_recursive(arr, pi + 1, high);
    }
}

fn partition(arr: &mut Vec<i32>, low: usize, high: usize) -> usize {
    let pivot = arr[high];
    let mut i = low;
    
    for j in low..high {
        if arr[j] < pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    arr.swap(i, high);
    i
}

pub struct SortContext {
    strategy: Box<dyn SortStrategy>,
}

impl SortContext {
    pub fn new(strategy: Box<dyn SortStrategy>) -> Self {
        SortContext { strategy }
    }

    pub fn set_strategy(&mut self, strategy: Box<dyn SortStrategy>) {
        self.strategy = strategy;
    }

    pub fn execute_sort(&self, data: &mut Vec<i32>) {
        self.strategy.sort(data);
    }
}

/// Command pattern for encapsulating requests.
pub trait Command {
    fn execute(&self);
    fn undo(&self);
}

pub struct Light {
    is_on: Arc<Mutex<bool>>,
}

impl Light {
    pub fn new() -> Self {
        Light {
            is_on: Arc::new(Mutex::new(false)),
        }
    }

    pub fn turn_on(&self) {
        let mut is_on = self.is_on.lock().unwrap();
        *is_on = true;
        println!("Light is ON");
    }

    pub fn turn_off(&self) {
        let mut is_on = self.is_on.lock().unwrap();
        *is_on = false;
        println!("Light is OFF");
    }

    pub fn is_on(&self) -> bool {
        *self.is_on.lock().unwrap()
    }
}

pub struct LightOnCommand {
    light: Arc<Light>,
}

impl LightOnCommand {
    pub fn new(light: Arc<Light>) -> Self {
        LightOnCommand { light }
    }
}

impl Command for LightOnCommand {
    fn execute(&self) {
        self.light.turn_on();
    }

    fn undo(&self) {
        self.light.turn_off();
    }
}

pub struct LightOffCommand {
    light: Arc<Light>,
}

impl LightOffCommand {
    pub fn new(light: Arc<Light>) -> Self {
        LightOffCommand { light }
    }
}

impl Command for LightOffCommand {
    fn execute(&self) {
        self.light.turn_off();
    }

    fn undo(&self) {
        self.light.turn_on();
    }
}

pub struct RemoteControl {
    command: Option<Box<dyn Command>>,
    last_command: Option<Box<dyn Command>>,
}

impl RemoteControl {
    pub fn new() -> Self {
        RemoteControl {
            command: None,
            last_command: None,
        }
    }

    pub fn set_command(&mut self, command: Box<dyn Command>) {
        self.command = Some(command);
    }

    pub fn press_button(&mut self) {
        if let Some(command) = self.command.take() {
            command.execute();
            self.last_command = Some(command);
        }
    }

    pub fn press_undo(&self) {
        if let Some(command) = &self.last_command {
            command.undo();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singleton() {
        let db1 = DatabaseConnection::get_instance();
        let db2 = DatabaseConnection::get_instance();
        
        // Both should be the same instance
        assert_eq!(
            db1 as *const DatabaseConnection,
            db2 as *const DatabaseConnection
        );
    }

    #[test]
    fn test_builder() {
        let computer = ComputerBuilder::new()
            .cpu("Intel i7".to_string())
            .ram(16)
            .storage(512)
            .gpu("NVIDIA RTX 4080".to_string())
            .wifi(true)
            .bluetooth(true)
            .build()
            .unwrap();

        assert_eq!(computer.cpu, "Intel i7");
        assert_eq!(computer.ram, 16);
        assert_eq!(computer.storage, 512);
        assert!(computer.wifi);
    }

    #[test]
    fn test_factory() {
        let car = VehicleFactory::create_vehicle("car", "Toyota".to_string()).unwrap();
        let motorcycle = VehicleFactory::create_vehicle("motorcycle", "Honda".to_string()).unwrap();

        assert_eq!(car.get_type(), "Car");
        assert_eq!(motorcycle.get_type(), "Motorcycle");
    }

    #[test]
    fn test_strategy() {
        let mut data = vec![64, 34, 25, 12, 22, 11, 90];
        let mut context = SortContext::new(Box::new(BubbleSort));
        
        context.execute_sort(&mut data);
        assert_eq!(data, vec![11, 12, 22, 25, 34, 64, 90]);
    }
}