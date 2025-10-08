package main

import (
    "fmt"
)

// Oop: Abstraction
// AI/ML Training Sample

type Abstraction struct {
    Data string
}

func NewAbstraction() *Abstraction {
    return &Abstraction{
        Data: "",
    }
}

func (s *Abstraction) Process(input string) {
    s.Data = input
}

func (s *Abstraction) Validate() bool {
    return len(s.Data) > 0
}

func (s *Abstraction) GetData() string {
    return s.Data
}

func main() {
    instance := NewAbstraction()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
