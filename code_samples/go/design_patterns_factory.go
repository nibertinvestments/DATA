package main

import (
    "fmt"
)

// Design Patterns: Factory
// AI/ML Training Sample

type Factory struct {
    Data string
}

func NewFactory() *Factory {
    return &Factory{
        Data: "",
    }
}

func (s *Factory) Process(input string) {
    s.Data = input
}

func (s *Factory) Validate() bool {
    return len(s.Data) > 0
}

func (s *Factory) GetData() string {
    return s.Data
}

func main() {
    instance := NewFactory()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
