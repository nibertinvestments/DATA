package main

import (
    "fmt"
)

// Functional: Monads
// AI/ML Training Sample

type Monads struct {
    Data string
}

func NewMonads() *Monads {
    return &Monads{
        Data: "",
    }
}

func (s *Monads) Process(input string) {
    s.Data = input
}

func (s *Monads) Validate() bool {
    return len(s.Data) > 0
}

func (s *Monads) GetData() string {
    return s.Data
}

func main() {
    instance := NewMonads()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
