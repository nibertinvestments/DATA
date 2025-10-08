package main

import (
    "fmt"
)

// Data Structures: Trie
// AI/ML Training Sample

type Trie struct {
    Data string
}

func NewTrie() *Trie {
    return &Trie{
        Data: "",
    }
}

func (s *Trie) Process(input string) {
    s.Data = input
}

func (s *Trie) Validate() bool {
    return len(s.Data) > 0
}

func (s *Trie) GetData() string {
    return s.Data
}

func main() {
    instance := NewTrie()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
