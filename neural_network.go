package main

import (
	"fmt"

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
