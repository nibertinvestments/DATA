package main

import (
    "fmt"
)

// Web Development: Rest Api
// AI/ML Training Sample

type RestApi struct {
    Data string
}

func NewRestApi() *RestApi {
    return &RestApi{
        Data: "",
    }
}

func (s *RestApi) Process(input string) {
    s.Data = input
}

func (s *RestApi) Validate() bool {
    return len(s.Data) > 0
}

func (s *RestApi) GetData() string {
    return s.Data
}

func main() {
    instance := NewRestApi()
    instance.Process("example")
    fmt.Printf("Data: %s\n", instance.GetData())
    fmt.Printf("Valid: %v\n", instance.Validate())
}
