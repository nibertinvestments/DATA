package main

import (
    "fmt"
)

// Error Handling: Error Propagation
// AI/ML Training Sample

type ErrorPropagation struct {
    Data string
}

func NewErrorPropagation() *ErrorPropagation {
    return &ErrorPropagation{
        Data: "",
    }
}

func (s *ErrorPropagation) Process(input string) {
    s.Data = input
}

func (s *ErrorPropagation) Validate() bool {
    return len(s.Data) > 0
}

func (s *ErrorPropagation) GetData() string {
    return s.Data
}

func main() {
    instance := NewErrorPropagation()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
