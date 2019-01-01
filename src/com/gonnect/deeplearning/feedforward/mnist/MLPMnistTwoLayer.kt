package com.gonnect.deeplearning.feedforward.mnist

import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.Logger
import org.slf4j.LoggerFactory


object MLPMnistTwoLayer {

    private val log = LoggerFactory.getLogger(MLPMnistTwoLayerExample::class.java)

    @Throws(Exception::class)
    @JvmStatic fun main(args: Array<String>) {
        //number of rows and columns in the input pictures
        val numRows = 28
        val numColumns = 28
        val outputNum = 10 // number of output classes
        val batchSize = 64 // batch size for each epoch
        val rngSeed = 123 // random number seed for reproducibility
        val numEpochs = 15 // number of epochs to perform
        val rate = 0.0015 // learning rate

        //Get the DataSetIterators:
        val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
        val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)


        log.info("Build model....")
        val conf = NeuralNetConfiguration.Builder()
                .seed(rngSeed.toLong()) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm

                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Nesterovs(rate, 0.98)) //specify the rate of change of the learning rate.
                .l2(rate * 0.005) // regularize learning model
                .list()
                .layer(0, DenseLayer.Builder() //create the first input layer.
                        .nIn(numRows * numColumns)
                        .nOut(500)
                        .build())
                .layer(1, DenseLayer.Builder() //create the second input layer
                        .nIn(500)
                        .nOut(100)
                        .build())
                .layer(2, OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .activation(Activation.SOFTMAX)
                        .nIn(100)
                        .nOut(outputNum)
                        .build())
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build()

        val model = MultiLayerNetwork(conf)
        model.init()
        model.setListeners(ScoreIterationListener(5))  //print the score with every iteration

        log.info("Train model....")
        for (i in 0..numEpochs - 1) {
            log.info("Epoch " + i)
            model.fit(mnistTrain)
        }


        log.info("Evaluate model....")
        val eval = Evaluation(outputNum) //create an evaluation object with 10 possible classes
        while (mnistTest.hasNext()) {
            val next = mnistTest.next()
            val output = model.output(next.features) //get the networks prediction
            eval.eval(next.getLabels(), output) //check the prediction against the true class
        }

        log.info(eval.stats())
        log.info("****************Example finished********************")
    }

}
