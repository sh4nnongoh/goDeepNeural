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

import "math"

type NeuralNetwork struct {
	Layer []*Layer
}

func (nn *NeuralNetwork) init(layerDim []int, aFunc []ActivationFunction) {
	/*
	*	Initialize the model with the given layerDim of size L
	*/
	var prevDim = 1
	for i, dim := range layerDim {
		nn.Layer = append(nn.Layer, NewLayer(dim, prevDim, aFunc[i]))
		prevDim = dim
	}
}

func (nn *NeuralNetwork) modelForwardPropogate(X *[]float64) *[]float64{
	/*
	*	Implement forward propagation
	*/
	for i,l := range nn.Layer {
		X = l.linearActivationForward(X)
	}
	return X
}

func (nn *NeuralNetwork) modelBackwardPropogate(A *[]float64) *[]float64{
	/*
	*	Implement Backward propagation
	*/
	for i,l := range nn.Layer {
		A = l.linearActivationBackward(A)
	}
	return A
}

func (nn *NeuralNetwork) computeCost(AL []float64, Y []float64) {
	/*
	*	Compares the calculated values against the correct values
	*/
}

func (nn *NeuralNetwork) updateParameters(learningRate float64) *[]float64{
	/*
	*	Update parameters using gradient descent
	*/
	for i,l := range nn.Layer {
		// update l.W base on l.AGrad and learningRate
	}
	// return all Ws
}

func (nn *NeuralNetwork) predict(X []float64, Y []float64) []float64{
	/*
	*	Perform prediction based on the current trained model nn
	*/
	//return p
}

func NewLayer(layerDimCurr, layerDimPrev int, aFunc ActivationFunction) *Layer {
	l := &Layer{}
	l.init(layerDimCurr, layerDimPrev, aFunc)
	return l
}

type Layer struct {
	//x []*float64 // input data of shape (input size, example size)
	W []*float64 // weights of each synapses of shape (neuron size, input size)
	b []*float64 // bias for each of the synapses (neuron size, 1)
	ActivationFunction ActivationFunction `json:"-"`
	A []*float64 // activation data of shape (1, input size)
	AGrad []*float64 // gradients of activation data of shape (1, input size)
}

func (l *Layer) init(layerDimCurr, layerDimPrev int, aFunc ActivationFunction) {
	/*
	*	Initialize the layer with a given number of neurons 
	*	that are initialized with random weights
	*/
	for i := range (layerDimCurr * layerDimPrev){
		l.W = append(l.W, rand.NormFloat64())
	}
	// weightMat = mat.NewDense(layerDimCurr, layerDimPrev, l.w)
	for i := range (layerDimCurr){
		l.b = append(l.b, 0)
	}
	// biasMat = mat.NewDense(layerDimCurr, 1, l.b)

	l.ActivationFunction = aFunc
}

func (l *Layer) linearActivationForward(APrev []float64) *[]float64 {
	/*
	*	Implement the linear part of a layer's foward propagation with Activation
	*/
	// var Z mat.Dense
	// Z.Mul(l.W,APrev)
	// Z.Add(Z,l.b)
	// l.A = l.Activate(Z)
	// return l.A
}

func (l *Layer) linearActivationBackward(dA []float64) *[]float64 {
	/*
	*	Implement the linear part of a layer's backward propagation with Activation
	*/
	// var Z mat.Dense
	// Z.Mul(l.W,APrev)
	// Z.Add(Z,l.b)
	// l.A = l.Activate(Z)
	// return l.A
}

func NewLogisticFunc(a float64) ActivationFunction {
	return func(x float64) float64 {
		return LogisticFunc(x, a)
	}
}
func LogisticFunc(x, a float64) float64 {
	return 1 / (1 + math.Exp(-a*x))
}

type ActivationFunction func(float64) float64