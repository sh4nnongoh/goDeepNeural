package main

/*

Input layer, Hidden Layer, Output layer
Each layer has activations a[0] = X, a[2] = {a1[2]
											 a2[2]
											 a3[2]
											 a4[2]}
Activations are the outputs of each layer l
z[l] = w[l] * x[l] + b
a[l] = activate(z[l])

If there are L layers than out layer y = a[L]

*/

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func MatrixSumKeepDims(m *mat.Dense) *mat.Dense {
	var row, col = m.Dims()
	sum := 0.0
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			sum += m.At(i, j)
		}
	}

	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			m.Set(i, j, sum)
		}
	}

	return m
}
func MatrixSum(m *mat.Dense) float64 {
	var row, col = m.Dims()
	sum := 0.0
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			sum += m.At(i, j)
		}
	}

	return sum
}

func main() {
	fmt.Println("Hello World!")

	// var Z mat.Dense
	// Z.Mul(l.W,APrev)
	// Z.Add(Z,l.b)
	// l.A = l.Activate(Z)

	// NewDense retuns a pointer to the new Mat object
	a := mat.NewDense(4, 3, []float64{
		1, 0, 0,
		1, 0, 1,
		0, 1, 1,
		1, 1, 1,
	})
	b := mat.NewDense(3, 1, []float64{
		0,
		0,
		1,
	})
	fmt.Println("a : ", mat.Formatted(a, mat.Prefix("    "), mat.Squeeze()))
	fmt.Println("b : ", mat.Formatted(b, mat.Prefix("    "), mat.Squeeze()))

	q := &a
	w := &b
	fmt.Println("a : ", mat.Formatted(*q, mat.Prefix("    "), mat.Squeeze()))
	fmt.Println("b : ", mat.Formatted(*w, mat.Prefix("    "), mat.Squeeze()))

	// This declaration return the actual object
	var c mat.Dense
	e := &c
	e.Mul(*q, *w)
	fmt.Println("e : ", mat.Formatted(e, mat.Prefix("    "), mat.Squeeze()))
	fmt.Println("c : ", mat.Formatted(&c, mat.Prefix("    "), mat.Squeeze()))
	var row, col = e.Dims()
	fmt.Println("dims(e): ", row, col)

	e.Scale(1/float64(row), e)
	fmt.Println("e : ", mat.Formatted(e, mat.Prefix("    "), mat.Squeeze()))

	fmt.Println("e sum : ", mat.Formatted(MatrixSumKeepDims(e), mat.Prefix("    "), mat.Squeeze()))

	fmt.Println("ReLU(0.9) : ", NewReLUFunc()(0, 0, -1))

	e.Apply(NewReLUFunc(), e)
	fmt.Println("e : ", mat.Formatted(e, mat.Prefix("    "), mat.Squeeze()))

	e.Apply(NewSigmoidFunc(), e)
	fmt.Println("e : ", mat.Formatted(e, mat.Prefix("    "), mat.Squeeze()))

	var layer = NewLayerRandomWeight(3, 3, NewReLUFunc())
	x := mat.NewDense(3, 1, []float64{
		1,
		1,
		1,
	})
	out := layer.linearActivationForward(x)
	fmt.Println("layer matrix out : ", mat.Formatted(out))

	e.Apply(func(i, j int, v float64) float64 { return math.Log(v) }, e)
	fmt.Println("test mat log : ", mat.Formatted(e))

	var test mat.Dense
	test.Mul(out, x.T())
	fmt.Println("test : ", mat.Formatted(&test))
	row, col = test.Dims()
	fmt.Println("test.Dims : ", row, col)
	fmt.Println("Dot Product : ", MatrixSum(&test))

	tmp1 := mat.NewDense(3, 3, make([]float64, 9))
	row, col = b.Dims()
	fmt.Println("b.Dims : ", row, col)
	row, col = x.Dims()
	fmt.Println("x.Dims : ", row, col)
	row, col = b.T().Dims()
	fmt.Println("b.T().Dims : ", row, col)
	//var row, _ = Y.Dims()
	x.Apply(func(i, j int, v float64) float64 { return math.Log(v) }, x)
	tmp1.Mul(x, b.T())
	fmt.Println("tmp1 : ", mat.Formatted(tmp1))
}
