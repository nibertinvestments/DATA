package main

import (
    "fmt"
)

// Error Handling: Custom Exceptions
// AI/ML Training Sample

type CustomExceptions struct {
    Data string
}

func NewCustomExceptions() *CustomExceptions {
    return &CustomExceptions{
        Data: "",
    }
}

func (s *CustomExceptions) Process(input string) {
    s.Data = input
}

func (s *CustomExceptions) Validate() bool {
    return len(s.Data) > 0
}

func (s *CustomExceptions) GetData() string {
    return s.Data
}

func main() {
    instance := NewCustomExceptions()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
