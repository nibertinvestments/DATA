package main

import (
    "fmt"
)

// Web Development: Middleware
// AI/ML Training Sample

type Middleware struct {
    Data string
}

func NewMiddleware() *Middleware {
    return &Middleware{
        Data: "",
    }
}

func (s *Middleware) Process(input string) {
    s.Data = input
}

func (s *Middleware) Validate() bool {
    return len(s.Data) > 0
}

func (s *Middleware) GetData() string {
    return s.Data
}

func main() {
    instance := NewMiddleware()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
