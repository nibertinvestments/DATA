package main

import (
    "fmt"
)

// Data Structures: Stack
// AI/ML Training Sample

type Stack struct {
    Data string
}

func NewStack() *Stack {
    return &Stack{
        Data: "",
    }
}

func (s *Stack) Process(input string) {
    s.Data = input
}

func (s *Stack) Validate() bool {
    return len(s.Data) > 0
}

func (s *Stack) GetData() string {
    return s.Data
}

func main() {
    instance := NewStack()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
