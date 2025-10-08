package main

import (
    "fmt"
)

// Functional: Closures
// AI/ML Training Sample

type Closures struct {
    Data string
}

func NewClosures() *Closures {
    return &Closures{
        Data: "",
    }
}

func (s *Closures) Process(input string) {
    s.Data = input
}

func (s *Closures) Validate() bool {
    return len(s.Data) > 0
}

func (s *Closures) GetData() string {
    return s.Data
}

func main() {
    instance := NewClosures()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
