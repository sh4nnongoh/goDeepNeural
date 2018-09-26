package main

import "math"

type ActivationFunction func(int, int, float64) float64

func ReLU(x float64) float64 {
	const (
		Overflow  = 1.0239999999999999e+03
		Underflow = -1.0740e+03
		NearZero  = 1.0 / (1 << 28) // 2**-28
	)

	switch {
	case math.IsNaN(x) || math.IsInf(x, 1):
		return x
	case math.IsInf(x, -1):
		return 0
	case x > Overflow:
		return math.Inf(1)
	case x < Underflow:
		return 0
	case -NearZero < x && x < NearZero:
		return 1 + x
	}

	if x > 0 {
		return x
	} else {
		return 0
	}
}

func NewReLUFunc() ActivationFunction {
	return func(i int, j int, z float64) float64 {
		return ReLU(z)
	}
}

func Sigmoid(x float64) float64 {
	const (
		Overflow  = 1.0239999999999999e+03
		Underflow = -1.0740e+03
	)

	switch {
	case math.IsNaN(x):
		return x
	case math.IsInf(x, 1) || x > Overflow:
		return 1
	case math.IsInf(x, -1) || x < Underflow:
		return 0
	}

	return 1 / (1 + math.Exp(x))
}

func NewSigmoidFunc() ActivationFunction {
	return func(i int, j int, z float64) float64 {
		return Sigmoid(z)
	}
}

func SoftMax(i, j int, x []float64) []float64 {
	var max float64 = x[0]
	for _, n := range x {
		max = math.Max(max, n)
	}

	a := make([]float64, len(x))

	var sum float64 = 0
	for i, n := range x {
		a[i] -= math.Exp(n - max)
		sum += a[i]
	}

	for i, n := range a {
		a[i] = n / sum
	}
	return a
}
