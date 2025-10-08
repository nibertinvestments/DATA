package main

import (
    "fmt"
)

// Data Structures: Linked List
// AI/ML Training Sample

type LinkedList struct {
    Data string
}

func NewLinkedList() *LinkedList {
    return &LinkedList{
        Data: "",
    }
}

func (s *LinkedList) Process(input string) {
    s.Data = input
}

func (s *LinkedList) Validate() bool {
    return len(s.Data) > 0
}

func (s *LinkedList) GetData() string {
    return s.Data
}

func main() {
    instance := NewLinkedList()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
