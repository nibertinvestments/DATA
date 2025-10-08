package main

import (
    "fmt"
)

// Algorithms: Searching
// AI/ML Training Sample

type Searching struct {
    Data string
}

func NewSearching() *Searching {
    return &Searching{
        Data: "",
    }
}

func (s *Searching) Process(input string) {
    s.Data = input
}

func (s *Searching) Validate() bool {
    return len(s.Data) > 0
}

func (s *Searching) GetData() string {
    return s.Data
}

func main() {
    instance := NewSearching()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
