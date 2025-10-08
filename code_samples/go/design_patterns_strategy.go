package main

import (
    "fmt"
)

// Design Patterns: Strategy
// AI/ML Training Sample

type Strategy struct {
    Data string
}

func NewStrategy() *Strategy {
    return &Strategy{
        Data: "",
    }
}

func (s *Strategy) Process(input string) {
    s.Data = input
}

func (s *Strategy) Validate() bool {
    return len(s.Data) > 0
}

func (s *Strategy) GetData() string {
    return s.Data
}

func main() {
    instance := NewStrategy()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
