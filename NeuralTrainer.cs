/*This file contains the most common method of training Neural Network which is through
 BackPropagation Algorithm*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkLibrary
{
    public class NeuralTrainer
    {
        private NeuralNetwork m_network;
        private int m_inputSize;
        private int[] m_layerSize;
        private int m_layerCount;
        private double[][] m_delta;

        private double[][] m_bias;
        private double[][] m_perviousBiasDelta;

        private double[][][] m_weight;
        private double[][][] m_previousWeightDelta;

        private TransferFunction[] m_transferFunction;
        private double[][] m_layerInput;
        private double[][] m_layerOutput;

        public NeuralTrainer(NeuralNetwork n)
        {
            //Assigning the values
            m_network = n;
            m_inputSize = m_network.GetInputSize();
            m_layerSize = m_network.GetLayerSize();
            m_layerCount = m_network.GetLayerCount();
            m_transferFunction = m_network.GetTransferFunctions();
            m_layerInput = m_network.GetLayerInput();
            m_delta = m_network.GetDelta();
            m_layerOutput = m_network.GetLayerOutput();
            m_bias = m_network.GetBias();
            m_weight = m_network.GetWeights();
            m_perviousBiasDelta = m_network.GetPreviousBias();
            m_previousWeightDelta = m_network.GetPreviousWeight();
        }
        public double BackPropagationTrain(ref double[] input, ref double[] desired, double TrainingRate, double Momentum)
        {
            


            // Parameter Validation
            if (input.Length != m_inputSize)
                throw new ArgumentException("Invalid input parameter", "input");
            if (desired.Length != m_layerSize[m_layerCount - 1])
                throw new ArgumentException("Invalid input parameter", "desired");

            // Local variable
            double error = 0.0, sum = 0.0, weightDelta = 0.0, biasDelta = 0.0;
            double[] output = new double[m_layerSize[m_layerCount - 1]];

            // Run the network
            m_network.Run(ref input, out output);

            // Back-propagate the error
            for (int l = m_layerCount - 1; l >= 0; l--)
            {
                // Output layer
                if (l == m_layerCount - 1)
                {
                    for (int k = 0; k < m_layerSize[l]; k++)
                    {
                        m_delta[l][k] = output[k] - desired[k];
                        error += Math.Pow(m_delta[l][k], 2);
                        m_delta[l][k] *= TransferFunctions.EvaluateDerivative(m_transferFunction[l],
                                                                            m_layerInput[l][k]);
                    }
                }
                else // Hidden layer
                {
                    for (int i = 0; i < m_layerSize[l]; i++)
                    {
                        sum = 0.0;
                        for (int j = 0; j < m_layerSize[l + 1]; j++)
                        {
                            sum += m_weight[l + 1][i][j] * m_delta[l + 1][j];
                        }
                        sum *= TransferFunctions.EvaluateDerivative(m_transferFunction[l], m_layerInput[l][i]);

                        m_delta[l][i] = sum;
                    }
                }
            }
            // Update the weights and biases
            for (int l = 0; l < m_layerCount; l++)
                for (int i = 0; i < (l == 0 ? m_inputSize : m_layerSize[l - 1]); i++)
                    for (int j = 0; j < m_layerSize[l]; j++)
                    {
                        
                        weightDelta = TrainingRate * m_delta[l][j] * (l == 0 ? input[i] : m_layerOutput[l - 1][i])
                                       + Momentum * m_previousWeightDelta[l][i][j];
                        m_weight[l][i][j] -= weightDelta;

                        m_previousWeightDelta[l][i][j] = weightDelta;
                    }

            for (int l = 0; l < m_layerCount; l++)
                for (int i = 0; i < m_layerSize[l]; i++)
                {
                    biasDelta = TrainingRate * m_delta[l][i];
                    m_bias[l][i] -= biasDelta + Momentum * m_perviousBiasDelta[l][i];

                    m_perviousBiasDelta[l][i] = biasDelta;
                }

            return error;
        }
    }
}
