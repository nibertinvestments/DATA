package main

import (
    "fmt"
)

// Functional: Currying
// AI/ML Training Sample

type Currying struct {
    Data string
}

func NewCurrying() *Currying {
    return &Currying{
        Data: "",
    }
}

func (s *Currying) Process(input string) {
    s.Data = input
}

func (s *Currying) Validate() bool {
    return len(s.Data) > 0
}

func (s *Currying) GetData() string {
    return s.Data
}

func main() {
    instance := NewCurrying()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
