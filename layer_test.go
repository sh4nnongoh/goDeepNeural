package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleHello() {
	fmt.Println("Hello World!")
	// Output: Hello World!
}

func ExampleLayer() {
	var layer = NewLayerRandomWeight(3, 3, NewReLUFunc())
	fmt.Println("layer matrix : ", mat.Formatted(layer.W))
	// Output:
	//⎡  -1.233758177597947  -0.12634751070237293   -0.5209945711531503⎤
	//⎢    2.28571911769958    0.3228052526115799    0.5900672875996937⎥
	//⎣ 0.15880774017643562    0.9892020842955818    -0.731283016177479⎦
}
func ExampleLayerActivation() {
	var layer = NewLayerRandomWeight(3, 3, NewReLUFunc())
	x := mat.NewDense(3, 1, []float64{
		0,
		0,
		1,
	})
	out := layer.linearActivationForward(x)
	fmt.Println("layer matrix : ", mat.Formatted(out))
	// Output:
	//⎡  -1.233758177597947  -0.12634751070237293   -0.5209945711531503⎤
	//⎢    2.28571911769958    0.3228052526115799    0.5900672875996937⎥
	//⎣ 0.15880774017643562    0.9892020842955818    -0.731283016177479⎦
}
