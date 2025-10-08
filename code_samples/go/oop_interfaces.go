package main

import (
    "fmt"
)

// Oop: Interfaces
// AI/ML Training Sample

type Interfaces struct {
    Data string
}

func NewInterfaces() *Interfaces {
    return &Interfaces{
        Data: "",
    }
}

func (s *Interfaces) Process(input string) {
    s.Data = input
}

func (s *Interfaces) Validate() bool {
    return len(s.Data) > 0
}

func (s *Interfaces) GetData() string {
    return s.Data
}

func main() {
    instance := NewInterfaces()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
