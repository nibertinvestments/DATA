package main

import (
    "fmt"
)

// Design Patterns: Observer
// AI/ML Training Sample

type Observer struct {
    Data string
}

func NewObserver() *Observer {
    return &Observer{
        Data: "",
    }
}

func (s *Observer) Process(input string) {
    s.Data = input
}

func (s *Observer) Validate() bool {
    return len(s.Data) > 0
}

func (s *Observer) GetData() string {
    return s.Data
}

func main() {
    instance := NewObserver()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
