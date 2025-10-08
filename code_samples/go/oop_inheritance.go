package main

import (
    "fmt"
)

// Oop: Inheritance
// AI/ML Training Sample

type Inheritance struct {
    Data string
}

func NewInheritance() *Inheritance {
    return &Inheritance{
        Data: "",
    }
}

func (s *Inheritance) Process(input string) {
    s.Data = input
}

func (s *Inheritance) Validate() bool {
    return len(s.Data) > 0
}

func (s *Inheritance) GetData() string {
    return s.Data
}

func main() {
    instance := NewInheritance()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
