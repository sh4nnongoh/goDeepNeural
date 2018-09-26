package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleNewNeuralNetwork() {
	nn := NewNeuralNetwork(3, []int{3, 4, 5, 4, 3}, []ActivationFunction{NewReLUFunc(), NewReLUFunc(), NewReLUFunc(), NewReLUFunc(), NewReLUFunc()})
	fmt.Println("Layer[0]: ", mat.Formatted(nn.Layer[0].W))
	fmt.Println("Layer[1]: ", mat.Formatted(nn.Layer[1].W))
	fmt.Println("Layer[2]: ", mat.Formatted(nn.Layer[2].W))
	fmt.Println("Layer[3]: ", mat.Formatted(nn.Layer[3].W))
	fmt.Println("Layer[4]: ", mat.Formatted(nn.Layer[4].W))
	fmt.Println("Output: ", mat.Formatted(nn.Layer[4].A))

	// Output: Something
}

func ExampleModelForward() {
	const (
		inputSize = 3
	)
	nn := NewNeuralNetwork(inputSize, []int{3, 4, 5, 4, 3}, []ActivationFunction{NewReLUFunc(), NewReLUFunc(), NewReLUFunc(), NewReLUFunc(), NewReLUFunc()})
	x := mat.NewDense(inputSize, 1, []float64{
		0,
		0,
		1,
	})
	nn.modelForwardPropogate(x)
	fmt.Println("Layer[0]: ", mat.Formatted(nn.Layer[0].W))
	fmt.Println("Layer[1]: ", mat.Formatted(nn.Layer[1].W))
	fmt.Println("Layer[2]: ", mat.Formatted(nn.Layer[2].W))
	fmt.Println("Layer[3]: ", mat.Formatted(nn.Layer[3].W))
	fmt.Println("Layer[4]: ", mat.Formatted(nn.Layer[4].W))
	fmt.Println("Output: ", mat.Formatted(nn.Layer[4].A))

	fmt.Println("NN Size: ", len(nn.Layer))

	// Output: Something
}
