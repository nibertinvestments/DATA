package main

import (
    "fmt"
)

// Design Patterns: Singleton
// AI/ML Training Sample

type Singleton struct {
    Data string
}

func NewSingleton() *Singleton {
    return &Singleton{
        Data: "",
    }
}

func (s *Singleton) Process(input string) {
    s.Data = input
}

func (s *Singleton) Validate() bool {
    return len(s.Data) > 0
}

func (s *Singleton) GetData() string {
    return s.Data
}

func main() {
    instance := NewSingleton()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
