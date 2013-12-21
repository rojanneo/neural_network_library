/*This file contains the basic elements of neural network creation*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;

namespace NeuralNetworkLibrary
{

    

    public class NeuralNetwork
    {
        #region Constructors

        public NeuralNetwork(int[] layerSizes, TransferFunction[] transferFunctions)
        {
            // Validate the input data
            if (transferFunctions.Length != layerSizes.Length || transferFunctions[0] != TransferFunction.None)
                throw new ArgumentException("Cannot construct a network with these parameters.");

            // Initialize network layers
            m_layerCount = layerSizes.Length - 1;
            m_inputSize = layerSizes[0];
            m_layerSize = new int[m_layerCount];

            for (int i = 0; i < m_layerCount; i++)
                m_layerSize[i] = layerSizes[i + 1];

            m_transferFunction = new TransferFunction[m_layerCount];
            for (int i = 0; i < m_layerCount; i++)
                m_transferFunction[i] = transferFunctions[i + 1];

            // Start dimensioning arrays
            m_bias = new double[m_layerCount][];
            m_perviousBiasDelta = new double[m_layerCount][];
            m_delta = new double[m_layerCount][];
            m_layerOutput = new double[m_layerCount][];
            m_layerInput = new double[m_layerCount][];

            m_weight = new double[m_layerCount][][];
            m_previousWeightDelta = new double[m_layerCount][][];

            // Fill 2 dimensional arrays
            for (int l = 0; l < m_layerCount; l++)
            {
                m_bias[l] = new double[m_layerSize[l]];
                m_perviousBiasDelta[l] = new double[m_layerSize[l]];
                m_delta[l] = new double[m_layerSize[l]];
                m_layerOutput[l] = new double[m_layerSize[l]];
                m_layerInput[l] = new double[m_layerSize[l]];

                m_weight[l] = new double[l == 0 ? m_inputSize : m_layerSize[l - 1]][];
                m_previousWeightDelta[l] = new double[l == 0 ? m_inputSize : m_layerSize[l - 1]][];

                for (int i = 0; i < (l == 0 ? m_inputSize : m_layerSize[l - 1]); i++)
                {
                    m_weight[l][i] = new double[m_layerSize[l]];
                    m_previousWeightDelta[l][i] = new double[m_layerSize[l]];
                }
            }

            // Initialize the weights
            for (int l = 0; l < m_layerCount; l++)
            {
                for (int j = 0; j < m_layerSize[l]; j++)
                {
                    m_bias[l][j] = Gaussian.GetRandomGaussian();
                    m_perviousBiasDelta[l][j] = 0.0;
                    m_layerOutput[l][j] = 0.0;
                    m_layerInput[l][j] = 0.0;
                    m_delta[l][j] = 0.0;
                }

                for (int i = 0; i < (l == 0 ? m_inputSize : m_layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < m_layerSize[l]; j++)
                    {
                        m_weight[l][i][j] = Gaussian.GetRandomGaussian();
                        m_previousWeightDelta[l][i][j] = 0.0;
                    }
                }
            }
        }

        public NeuralNetwork(string FilePath)
        {
            m_loaded = false;

            Load(FilePath);

            m_loaded = true;
        }

        #endregion

        #region Methods

        // Public methods
        public void Run(ref double[] input, out double[] output)
        {
            // Make sure we have enough data
            if (input.Length != m_inputSize)
                throw new ArgumentException("Input data is not of the correct dimension.");

            // Dimension
            output = new double[m_layerSize[m_layerCount - 1]];

            /* Run the network! */
            for (int l = 0; l < m_layerCount; l++)
            {
                for (int j = 0; j < m_layerSize[l]; j++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < (l == 0 ? m_inputSize : m_layerSize[l - 1]); i++)
                        sum += m_weight[l][i][j] * (l == 0 ? input[i] : m_layerOutput[l - 1][i]);

                    sum += m_bias[l][j];
                    m_layerInput[l][j] = sum;

                    m_layerOutput[l][j] = TransferFunctions.Evaluate(m_transferFunction[l], sum);
                }
            }

            // Copy the output to the output array
            for (int i = 0; i < m_layerSize[m_layerCount - 1]; i++)
                output[i] = m_layerOutput[m_layerCount - 1][i];
        }

        

        public int GetInputSize()
        {
            return m_inputSize;
        }

        public void Save(string FilePath)
        {
            if (FilePath == null)
                return;
            XmlWriterSettings settings = new XmlWriterSettings();
            settings.Indent = true;
            settings.IndentChars = "\t";

            XmlWriter writer = XmlWriter.Create(FilePath, settings);


            // Begin document
            writer.WriteStartElement("NeuralNetwork");
            writer.WriteAttributeString("Type", "BackPropagation");

            // Parameters element
            writer.WriteStartElement("Parameters");

            writer.WriteElementString("Name", m_name);
            writer.WriteElementString("inputSize", m_inputSize.ToString());
            writer.WriteElementString("layerCount", m_layerCount.ToString());

            // Layer sizes
            writer.WriteStartElement("Layers");

            for (int l = 0; l < m_layerCount; l++)
            {
                writer.WriteStartElement("Layer");

                writer.WriteAttributeString("Index", l.ToString());
                writer.WriteAttributeString("Size", m_layerSize[l].ToString());
                writer.WriteAttributeString("Type", m_transferFunction[l].ToString());

                writer.WriteEndElement();	// Layer
            }

            writer.WriteEndElement();	// Layers

            writer.WriteEndElement();	// Parameters

            // Weights and biases
            writer.WriteStartElement("Weights");

            for (int l = 0; l < m_layerCount; l++)
            {
                writer.WriteStartElement("Layer");
                writer.WriteAttributeString("Index", l.ToString());

                for (int j = 0; j < m_layerSize[l]; j++)
                {
                    writer.WriteStartElement("Node");
                    writer.WriteAttributeString("Index", j.ToString());
                    writer.WriteAttributeString("Bias", m_bias[l][j].ToString());

                    for (int i = 0; i < (l == 0 ? m_inputSize : m_layerSize[l - 1]); i++)
                    {
                        writer.WriteStartElement("Axon");
                        writer.WriteAttributeString("Index", i.ToString());

                        writer.WriteString(m_weight[l][i][j].ToString());

                        writer.WriteEndElement();	// Axon
                    }

                    writer.WriteEndElement();	// Node
                }

                writer.WriteEndElement();	// Layer
            }

            writer.WriteEndElement();	// Weights

            writer.WriteEndElement();	// NeuralNetwork

            writer.Flush();
            writer.Close();
        }

        public void Load(string FilePath)
        {
            if (FilePath == null)
                return;

            m_doc = new XmlDocument();
            m_doc.Load(FilePath);

            string BasePath = "", NodePath = "";
            double value;

            // Load from xml
            if (xPathValue("NeuralNetwork/@Type") != "BackPropagation")
                return;

            BasePath = "NeuralNetwork/Parameters/";
            m_name = xPathValue(BasePath + "Name");

            int.TryParse(xPathValue(BasePath + "inputSize"), out m_inputSize);
            int.TryParse(xPathValue(BasePath + "layerCount"), out m_layerCount);

            m_layerSize = new int[m_layerCount];
            m_transferFunction = new TransferFunction[m_layerCount];

            BasePath = "NeuralNetwork/Parameters/Layers/Layer";
            for (int l = 0; l < m_layerCount; l++)
            {
                int.TryParse(xPathValue(BasePath + "[@Index='" + l.ToString() + "']/@Size"), out m_layerSize[l]);
                Enum.TryParse<TransferFunction>(xPathValue(BasePath + "[@Index='" + l.ToString() + "']/@Type"), out m_transferFunction[l]);
            }

            // Parse the Weights element

            // Start dimensioning arrays
            m_bias = new double[m_layerCount][];
            m_perviousBiasDelta = new double[m_layerCount][];
            m_delta = new double[m_layerCount][];
            m_layerOutput = new double[m_layerCount][];
            m_layerInput = new double[m_layerCount][];

            m_weight = new double[m_layerCount][][];
            m_previousWeightDelta = new double[m_layerCount][][];

            // Fill 2 dimensional arrays
            for (int l = 0; l < m_layerCount; l++)
            {
                m_bias[l] = new double[m_layerSize[l]];
                m_perviousBiasDelta[l] = new double[m_layerSize[l]];
                m_delta[l] = new double[m_layerSize[l]];
                m_layerOutput[l] = new double[m_layerSize[l]];
                m_layerInput[l] = new double[m_layerSize[l]];

                m_weight[l] = new double[l == 0 ? m_inputSize : m_layerSize[l - 1]][];
                m_previousWeightDelta[l] = new double[l == 0 ? m_inputSize : m_layerSize[l - 1]][];

                for (int i = 0; i < (l == 0 ? m_inputSize : m_layerSize[l - 1]); i++)
                {
                    m_weight[l][i] = new double[m_layerSize[l]];
                    m_previousWeightDelta[l][i] = new double[m_layerSize[l]];
                }
            }

            // Initialize the weights
            for (int l = 0; l < m_layerCount; l++)
            {
                BasePath = "NeuralNetwork/Weights/Layer[@Index='" + l.ToString() + "']/";
                for (int j = 0; j < m_layerSize[l]; j++)
                {
                    NodePath = "Node[@Index='" + j.ToString() + "']/@Bias";
                    double.TryParse(xPathValue(BasePath + NodePath), out value);

                    m_bias[l][j] = value;
                    m_perviousBiasDelta[l][j] = 0.0;
                    m_layerOutput[l][j] = 0.0;
                    m_layerInput[l][j] = 0.0;
                    m_delta[l][j] = 0.0;
                }

                for (int i = 0; i < (l == 0 ? m_inputSize : m_layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < m_layerSize[l]; j++)
                    {
                        NodePath = "Node[@Index='" + j.ToString() + "']/Axon[@Index='" + i.ToString() + "']";
                        double.TryParse(xPathValue(BasePath + NodePath), out value);

                        m_weight[l][i][j] = value;
                        m_previousWeightDelta[l][i][j] = 0.0;
                    }
                }
            }

            // "release"
            m_doc = null;
        }

        public void Nudge(double scalar)
        {
            // Go through all of the weights and biases and augment them
            for (int l = 0; l < m_layerCount; l++)
            {
                for (int j = 0; j < m_layerSize[l]; j++)
                {
                    // Nudge the weights
                    for (int i = 0; i < (l == 0 ? m_inputSize : m_layerSize[l - 1]); i++)
                    {
                        double w = m_weight[l][i][j];
                        double u = Gaussian.GetRandomGaussian(0f, w * scalar);
                        m_weight[l][i][j] += u;
                        m_previousWeightDelta[l][i][j] = 0f;
                    }

                    // Nudge the bias
                    double b = m_bias[l][j];
                    double v = Gaussian.GetRandomGaussian(0f, b * scalar);
                    m_bias[l][j] += v;
                    m_perviousBiasDelta[l][j] = 0f;
                }
            }
        }

        // Private methods
        private string xPathValue(string xPath)
        {
            XmlNode node = m_doc.SelectSingleNode(xPath);

            if (node == null)
                throw new ArgumentException("Cannot find specified node", xPath);

            return node.InnerText;
        }


        public int[] GetLayerSize()
        {
            return m_layerSize;
        }
        public int GetLayerCount()
        {
            return m_layerCount;
        }
        public TransferFunction[] GetTransferFunctions()
        {
            return m_transferFunction;
        }
        public double[][] GetDelta()
        {
            return m_delta;
        }

        public double[][] GetLayerInput()
        {
            return m_layerInput;
        }
        public double[][] GetLayerOutput()
        {
            return m_layerOutput;
        }
        public double[][] GetBias()
        {
            return m_bias;
        }
        public double[][] GetPreviousBias()
        {
            return m_perviousBiasDelta;
        }
        public double[][][] GetWeights()
        {
            return m_weight;
        }
        public double[][][] GetPreviousWeight()
        {
            return m_previousWeightDelta;
        }
        #endregion

        #region Public data

        public string m_name = "Default";

        #endregion

        #region Private data

        private int m_layerCount; // Number of layers
        private int m_inputSize; // Number of inputs that is to be provided to the network
        private int[] m_layerSize; // Number of nodes in layers
        private TransferFunction[] m_transferFunction; // Transfer functions for each layer
        
        private double[][] m_layerOutput; // Output of each layers
        private double[][] m_layerInput; // Input for each layer
        private double[][] m_bias; // Bias for each layer
        private double[][] m_delta; 
        private double[][] m_perviousBiasDelta;

        private double[][][] m_weight; // Weights for each connection
        private double[][][] m_previousWeightDelta;

        private XmlDocument m_doc = null;
        private bool m_loaded = true;

        #endregion
    }


}
