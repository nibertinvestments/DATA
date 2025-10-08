package main

import (
    "fmt"
)

// Oop: Encapsulation
// AI/ML Training Sample

type Encapsulation struct {
    Data string
}

func NewEncapsulation() *Encapsulation {
    return &Encapsulation{
        Data: "",
    }
}

func (s *Encapsulation) Process(input string) {
    s.Data = input
}

func (s *Encapsulation) Validate() bool {
    return len(s.Data) > 0
}

func (s *Encapsulation) GetData() string {
    return s.Data
}

func main() {
    instance := NewEncapsulation()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
