package main

import (
    "fmt"
)

// Web Development: Validation
// AI/ML Training Sample

type Validation struct {
    Data string
}

func NewValidation() *Validation {
    return &Validation{
        Data: "",
    }
}

func (s *Validation) Process(input string) {
    s.Data = input
}

func (s *Validation) Validate() bool {
    return len(s.Data) > 0
}

func (s *Validation) GetData() string {
    return s.Data
}

func main() {
    instance := NewValidation()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
