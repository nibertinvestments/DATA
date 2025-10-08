package main

import (
    "fmt"
)

// Oop: Polymorphism
// AI/ML Training Sample

type Polymorphism struct {
    Data string
}

func NewPolymorphism() *Polymorphism {
    return &Polymorphism{
        Data: "",
    }
}

func (s *Polymorphism) Process(input string) {
    s.Data = input
}

func (s *Polymorphism) Validate() bool {
    return len(s.Data) > 0
}

func (s *Polymorphism) GetData() string {
    return s.Data
}

func main() {
    instance := NewPolymorphism()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
