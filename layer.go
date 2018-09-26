package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func NewLayerRandomWeight(layerDimCurr, layerDimPrev int, aFunc ActivationFunction) *Layer {
	l := &Layer{}

	matrixSize := layerDimCurr * layerDimPrev
	w := make([]float64, matrixSize)
	for i := 0; i < matrixSize; i++ {
		w[i] = rand.NormFloat64()
	}
	W := mat.NewDense(layerDimCurr, layerDimPrev, w)
	b := mat.NewDense(layerDimCurr, 1, nil)
	A := mat.NewDense(layerDimCurr, 1, nil)
	AGrad := mat.NewDense(layerDimCurr, 1, nil)

	l.load(W, b, A, AGrad, aFunc)
	return l
}

type Layer struct {
	// Each layer has a set of neurons represented in matrixes of their associated inputs, weights, biases, and outputs.
	// Each layer will have the same Activation Function

	W *mat.Dense // weights of each synapses of shape (neuron size, input size)
	b *mat.Dense // bias for each of the synapses (neuron size, 1)

	ActivateFunc ActivationFunction
	A            *mat.Dense // current activations of shape (1, neuron size)
	APrev        *mat.Dense // activations from the previous layer (1, input size)
	AGrad        *mat.Dense // gradients of activation data of shape (1, neuron size)

	dW     *mat.Dense // weights of each synapses of shape (neuron size, input size)
	db     *mat.Dense // bias for each of the synapses (neuron size, 1)
	dAPrev *mat.Dense // activations from the previous layer (1, input size)
}

func (l *Layer) load(W, b, A, AGrad *mat.Dense, aFunc ActivationFunction) {
	/*
	*	Initialize the layer with the given variables
	 */
	l.W = W
	l.b = b
	l.A = A
	l.AGrad = AGrad

	l.ActivateFunc = aFunc
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
	l.A.Mul(l.W, APrev)
	//fmt.Println("layer A after MUL : ", mat.Formatted(l.A))
	l.A.Add(l.A, l.b)
	//fmt.Println("layer A after ADD : ", mat.Formatted(l.A))
	l.A.Apply(l.ActivateFunc, l.A)
	//fmt.Println("layer A after ACT : ", mat.Formatted(l.A))
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
