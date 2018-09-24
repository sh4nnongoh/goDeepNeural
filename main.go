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
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func ImportNeuralNetwork(weightsFile string) *NeuralNetwork {
	//
	//
	nn := &NeuralNetwork{}

	// Convert WeightsFile into an Array -> Matrix

	// Generate the Layers

	return nn
}

func NewNeuralNetwork(inputSize int, layerDim []int, aFunc []ActivationFunction) *NeuralNetwork {
	//
	//
	nn := &NeuralNetwork{}
	nn.init(inputSize, layerDim, aFunc)
	return nn
}

type NeuralNetwork struct {
	Layer []*Layer
}

func (nn *NeuralNetwork) init(inputSize int, layerDim []int, aFunc []ActivationFunction) {
	/*
	*	Initialize the model with the given layerDim of size L
	 */
	var prevDim = inputSize
	for i, dim := range layerDim {
		nn.Layer = append(nn.Layer, NewLayer(dim, prevDim, aFunc[i]))
		prevDim = dim
	}
}

func (nn *NeuralNetwork) modelForwardPropogate(X *mat.Dense) *mat.Dense {
	/*
	*	Implement forward propagation
	 */
	nn.Layer[0].linearActivationForward(X)
	for i, l := range nn.Layer[1:] {
		l.linearActivationForward(nn.Layer[i-1].A)
	}
	return nn.Layer[len(nn.Layer)-1].A
}

func (nn *NeuralNetwork) modelBackwardPropogate(A *mat.Dense) *mat.Dense {
	/*
	*	Implement Backward propagation
	 */
	for _, l := range nn.Layer {
		A = l.linearActivationBackward(A)
	}
	return A
}

func (nn *NeuralNetwork) computeCost(AL *mat.Dense, Y *mat.Dense) {
	/*
	*	Compares the calculated values against the correct values
	 */
}

func (nn *NeuralNetwork) updateParameters(learningRate float64) *mat.Dense {
	/*
	*	Update parameters using gradient descent
	 */
	for _, l := range nn.Layer {
		// update l.W base on l.AGrad and learningRate
		fmt.Println(l.W)
	}
	// return all Ws
	return nil
}

func (nn *NeuralNetwork) predict(X *mat.Dense, Y *mat.Dense) *mat.Dense {
	/*
	*	Perform prediction based on the current trained model nn
	 */
	//return p
	return nil
}

func NewLayer(layerDimCurr, layerDimPrev int, aFunc ActivationFunction) *Layer {
	l := &Layer{}

	matrixSize := layerDimCurr * layerDimPrev
	w := make([]float64, matrixSize)
	for i := 0; i < matrixSize; i++ {
		w[i] = rand.NormFloat64()
	}
	W := mat.NewDense(layerDimCurr, layerDimPrev, w)
	b := mat.NewDense(layerDimCurr, layerDimPrev, nil)
	A := mat.NewDense(layerDimCurr, 1, nil)
	AGrad := mat.NewDense(layerDimCurr, 1, nil)

	l.init(W, b, A, AGrad, aFunc)
	return l
}

type Layer struct {
	//x []*float64 // input data of shape (input size, example size)
	W *mat.Dense // weights of each synapses of shape (neuron size, input size)
	b *mat.Dense // bias for each of the synapses (neuron size, 1)

	ActivationFunction ActivationFunction `json:"-"`
	A                  *mat.Dense         // current activations of shape (1, neuron size)
	APrev              *mat.Dense         // activations from the previous layer (1, input size)
	AGrad              *mat.Dense         // gradients of activation data of shape (1, neuron size)

	dW     *mat.Dense // weights of each synapses of shape (neuron size, input size)
	db     *mat.Dense // bias for each of the synapses (neuron size, 1)
	dAPrev *mat.Dense // activations from the previous layer (1, input size)
}

func (l *Layer) init(W, b, A, AGrad *mat.Dense, aFunc ActivationFunction) {
	/*
	*	Initialize the layer with the given variables
	 */
	l.W = W
	l.b = b
	l.A = A
	l.AGrad = AGrad

	l.ActivationFunction = aFunc
}

func (l *Layer) linearActivationForward(APrev *mat.Dense) *mat.Dense {
	/*
	*	Implement the linear part of a layer's foward propagation with Activation
	 */
	// var Z mat.Dense
	// Z.Mul(l.W,APrev)
	// Z.Add(Z,l.b)
	// l.A = l.Activate(Z)
	// return l.A
	l.APrev = APrev
	l.A.Product(l.W, APrev)
	l.A.Add(l.A, l.b)
	// l.A.Apply(l.ActivationFunction, l.A)
	return l.A
}

func (l *Layer) linearActivationBackward(dZ *mat.Dense) *mat.Dense {
	/*
	*	Implement the linear part of a layer's backward propagation with Activation
	 */

	// Dense does not have ScaleVec
	// scale := 1.0 / l.APrev.Len()
	// dW := mat.ScaleVec(scale, mat.Dot(dA, l.APrev))
	// db := mat.ScaleVec(scale, mat.Sum(dA))
	// dA_prev = np.dot(W.T,dZ)
	var row, _ = l.APrev.Dims()
	l.dW.Mul(dZ, l.APrev.T())
	l.dW.Scale(1/float64(row), l.dW)
	MatrixSumKeepDims(l.db)
	l.db.Scale(1/float64(row), l.db)
	l.dAPrev.Mul(l.W.T(), dZ)
	return nil
}

func ReLU(i, j int, x float64) float64 {
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

func Sigmoid(i, j int, x float64) float64 {
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

type ActivationFunction func(int, int, float64) float64

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

}
