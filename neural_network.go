package main

import (
	"errors"
	"fmt"
	"math"

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
	nn.loadNNRandomWeight(inputSize, layerDim, aFunc)
	return nn
}

type NeuralNetwork struct {
	L     int // number of layers
	Layer []*Layer
}

func (nn *NeuralNetwork) loadNNRandomWeight(inputSize int, layerDim []int, aFunc []ActivationFunction) {
	/*
	*	Initialize the model with the given layerDim of size L
	 */
	var prevDim = inputSize
	for i, dim := range layerDim {
		nn.Layer = append(nn.Layer, NewLayerRandomWeight(dim, prevDim, aFunc[i]))
		prevDim = dim
	}
}

func (nn *NeuralNetwork) modelForwardPropogate(X *mat.Dense) *mat.Dense {
	/*
	*	Implement forward propagation
	 */
	nn.Layer[0].linearActivationForward(X)
	for i, l := range nn.Layer[1:] {
		l.linearActivationForward(nn.Layer[i].A)
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

func (nn *NeuralNetwork) computeCost(AL *mat.Dense, Y *mat.Dense) (*mat.Dense, error) {
	/*
	*	Compares the calculated values against the correct values
	 */
	rowAL, colAL := AL.Dims()
	rowY, colY := Y.Dims()
	if (rowAL != rowY) && (colAL != colY) {
		return nil, errors.New("Dimensions of inputs not equal!")
	}
	// cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
	var tmp1, tmp2, tmp3 mat.Dense
	tmp1.Apply(func(i, j int, v float64) float64 { return math.Log(v) }, AL)
	tmp1.Mul(&tmp1, Y.T())
	tmp1.Scale(-1/float64(rowAL), &tmp1)
	tmp2.Apply(func(i, j int, v float64) float64 { return 1 - v }, AL)
	tmp2.Apply(func(i, j int, v float64) float64 { return math.Log(v) }, &tmp2)
	tmp3.Apply(func(i, j int, v float64) float64 { return 1 - v }, Y)
	tmp1.Mul(&tmp3, tmp2.T())
	//cost := (-1.0/float64(row))*MatrixSum(&tmp1) - MatrixSum(&tmp2)

	return &tmp1, nil
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
