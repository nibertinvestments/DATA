package main

import (
    "fmt"
)

// Functional: Higher Order
// AI/ML Training Sample

type HigherOrder struct {
    Data string
}

func NewHigherOrder() *HigherOrder {
    return &HigherOrder{
        Data: "",
    }
}

func (s *HigherOrder) Process(input string) {
    s.Data = input
}

func (s *HigherOrder) Validate() bool {
    return len(s.Data) > 0
}

func (s *HigherOrder) GetData() string {
    return s.Data
}

func main() {
    instance := NewHigherOrder()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
