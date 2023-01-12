#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;
typedef vector<vector<vector<double>>> matrix;
typedef vector<vector<double>> layer;

/*------------------------------------- Helper Functions  --------------------------------*/

layer cross_correlate(layer input1, layer input2)
{
    layer output(input1.size() - input2.size() + 1, vector<double>(input1[0].size() - input2.size() + 1, 0));
    for (int i = 0; i < input1.size() - input2.size() + 1; i++)
    {
        for (int j = 0; j < input1[0].size() - input2.size() + 1; j++)
        {
            double output_term = 0.0;
            for (int k = 0; k < input2.size(); k++)
            {
                for (int l = 0; l < input2.size(); l++)
                {
                    output_term += input1[k + i][l + j] * input2[k][l];
                }
            }
            output[i][j] = output_term;
        }
    }
    return output;
}

layer cross_correlate3d(matrix input1, matrix input2)
{
    layer output(input1[0].size() - input2[0].size() + 1, vector<double>(input1[0][0].size() - input2[0].size() + 1, 0));
    for (int d = 0; d < input1.size(); d++)
    {
        layer res = cross_correlate(input1[d], input2[d]);
        for (int i = 0; i < input1[0].size() - input2[0].size() + 1; i++)
        {
            for (int j = 0; j < input1[0][0].size() - input2[0].size() + 1; j++)
            {
                output[i][j] += res[i][j];
            }
        }
    }
    return output;
}

void rotatematrix(layer &matrix)
{
    // Loop through each row of the matrix
    for (int i = 0; i < matrix.size(); i++)
    {
        // Reverse the elements in each row
        reverse(matrix[i].begin(), matrix[i].end());
    }

    // Reverse the order of the rows
    reverse(matrix.begin(), matrix.end());
}

layer full_convolve(layer input1, layer input2)
{

    rotatematrix(input2);

    layer padded_input1(input1.size() + 2 * (input2.size() - 1), vector<double>(input1[0].size() + 2 * (input2.size() - 1), 0));
    layer output(input1.size() + input2.size() - 1, vector<double>(input1[0].size() + input2.size() - 1, 0));
    for (int i = input2.size() - 1; i < input1.size() + input2.size() - 1; i++)
    {
        for (int j = input2.size() - 1; j < input1[0].size() + input2.size() - 1; j++)
        {
            padded_input1[i][j] = input1[i - input2.size() + 1][j - input2.size() + 1];
        }
    }
    output = cross_correlate(padded_input1, input2);
    return output;
}

vector<double> dot_product(layer weights, vector<double> input)
{
    vector<double> output;

    for (int i = 0; i < weights.size(); i++)
    {
        double output_term = 0.0;
        for (int j = 0; j < weights[0].size(); j++)
        {
            output_term += weights[i][j] * input[j];
        }
        output.push_back(output_term);
    }
    return output;
}

layer transpose(layer mat)
{
    layer output = layer(mat[0].size(), vector<double>(mat.size(), 0));

    for (int i = 0; i < output.size(); i++)
    {
        for (int j = 0; j < output[0].size(); j++)
        {
            output[i][j] = mat[j][i];
        }
    }
    return output;
}

layer matrix_multiplication(layer input1, layer input2)
{
    input2 = transpose(input2);
    layer output;

    for (int i = 0; i < input2.size(); i++)
    {
        output.push_back(dot_product(input1, input2[i]));
    }

    output = transpose(output);

    return output;
}

/*------------------------------------------------------------------------------------  Convolutional Layer ------------------------------------------------------------------------------------ */

class Convolutional
{
private:
    void random_initialize()
    {
        for (int i = 0; i < this->kernels.size(); i++)
        {
            for (int j = 0; j < this->kernels[i].size(); j++)
            {
                for (int k = 0; k < this->kernels[i][j].size(); k++)
                {
                    for (int l = 0; l < this->kernels[i][j][k].size(); l++)
                    {
                        this->kernels[i][j][k][l] = (float)(rand() / RAND_MAX);
                    }
                }
            }
        }

        for (int i = 0; i < this->biases.size(); i++)
        {
            for (int j = 0; j < this->biases[i].size(); j++)
            {
                for (int k = 0; k < this->biases[i][j].size(); k++)
                {

                    this->biases[i][j][k] = (float)(rand() / RAND_MAX);
                }
            }
        }
    }
    void Update_kernels(double learning_rate, vector<matrix> &kernel_gradient)
    {
        for (int i = 0; i < this->kernels.size(); i++)
        {
            for (int j = 0; j < this->kernels[i].size(); j++)
            {
                for (int k = 0; k < this->kernels[i][j].size(); k++)
                {
                    for (int l = 0; l < this->kernels[i][j][k].size(); l++)
                    {
                        this->kernels[i][j][k][l] -= learning_rate * kernel_gradient[i][j][k][l];
                    }
                }
            }
        }
    }
    void Update_biases(double learning_rate, matrix &deriv_output)
    {
        for (int i = 0; i < this->biases.size(); i++)
        {
            for (int j = 0; j < this->biases[i].size(); j++)
            {
                for (int k = 0; k < this->biases[i][j].size(); k++)
                {
                    this->biases[i][j][k] -= learning_rate * deriv_output[i][j][k];
                }
            }
        }
    }

public:
    int input_height, input_width, input_depth;
    int padding;
    int output_height, output_width, output_depth;
    int num_kernels;
    int kernel_dims;
    vector<matrix> kernels; // Kernels
    matrix biases;
    matrix output;
    matrix input;

    /* Constructor */
    Convolutional(vector<int> input_shape, int num_kernels, int kernel_dims, bool padd)
    {
        if (padd)
        {
            this->padding = (kernel_dims - 1) / 2;
        }
        else
        {
            this->padding = 0;
        }
        input_height = input_shape[0]; //{height,width,depth}
        input_width = input_shape[1];
        input_depth = input_shape[2];
        this->num_kernels = num_kernels;
        this->kernel_dims = kernel_dims;

        int reduction = 2 * (this->padding) + 1 - kernel_dims;
        output_height = input_height + reduction;
        output_width = input_width + reduction;
        output_depth = num_kernels;

        kernels = vector<matrix>(num_kernels, matrix(input_depth, vector<vector<double>>(kernel_dims, vector<double>(kernel_dims, 0))));
        biases = matrix(output_depth, vector<vector<double>>(output_height, vector<double>(output_width, 0)));
        output = matrix(output_depth, vector<vector<double>>(output_height, vector<double>(output_width, 0)));

        random_initialize();
    }

    matrix forward(matrix input)
    {

        this->input = input;
        int depth = output.size();

        for (int i = 0; i < depth; i++)
        {
            output[i] = cross_correlate3d(input, kernels[i]);
        }

        for (int i = 0; i < depth; i++)
        {

            for (int j = 0; j < output[i].size(); j++)
            {
                for (int k = 0; k < output[i][j].size(); k++)
                {
                    output[i][j][k] += biases[i][j][k];
                }
            }
        }
        return output;
    }

    matrix backward(matrix deriv_output, double learning_rate)
    {

        matrix input_gradient;
        vector<matrix> kernel_gradient;
        kernel_gradient = vector<matrix>(num_kernels, matrix(input_depth, vector<vector<double>>(kernel_dims, vector<double>(kernel_dims, 0))));
        input_gradient = matrix(input_depth, vector<vector<double>>(input_height, vector<double>(input_width)));

        for (int i = 0; i < output_depth; i++)
        {
            for (int j = 0; j < input_depth; j++)
            {
                kernel_gradient[i][j] = cross_correlate(input[j], deriv_output[i]);

                vector<vector<double>> temp = full_convolve(deriv_output[i], kernels[i][j]);

                for (int k = 0; k < input_gradient[j].size(); k++)
                {
                    for (int m = 0; m < input_gradient[j][k].size(); m++)
                    {
                        input_gradient[j][k][m] += temp[k][m];
                    }
                }
            }
        }

        Update_kernels(learning_rate, kernel_gradient);
        Update_biases(learning_rate, deriv_output);

        return input_gradient;
    }
};

/* ------------------------------------------------------------------------------- Activations Layer ------------------------------------------------------------------------------- */
class Activation_Layer
{
public:
    virtual matrix forward(matrix input) = 0;
    virtual matrix backward(matrix deriv_outputient, double learning_rate) = 0;
};

class ReLU : public Activation_Layer
{
public:
    matrix input;
    matrix forward(matrix input)
    {

        this->input = input;

        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < input[0].size(); j++)
            {
                for (int k = 0; k < input[0][0].size(); k++)
                {
                    input[i][j][k] = input[i][j][k] > 0 ? input[i][j][k] : (0.03 * input[i][j][k]);
                }
            }
        }
        return input;
    }

    matrix backward(matrix deriv_output, double learning_rate)
    {

        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < input[0].size(); j++)
            {
                for (int k = 0; k < input[0][0].size(); k++)
                {
                    input[i][j][k] = input[i][j][k] > 0 ? 1 : (0.03);
                }
            }
        }

        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < input[0].size(); j++)
            {
                for (int k = 0; k < input[0][0].size(); k++)
                {
                    deriv_output[i][j][k] *= input[i][j][k];
                }
            }
        }

        return deriv_output;
    }
};

class Tanh : public Activation_Layer
{
public:
    matrix input;
    matrix forward(matrix input)
    {

        this->input = input;

        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < input[0].size(); j++)
            {
                for (int k = 0; k < input[0][0].size(); k++)
                {
                    input[i][j][k] = tanh(input[i][j][k]);
                }
            }
        }
        return input;
    }

    matrix backward(matrix deriv_output, double learning_rate)
    {

        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < input[0].size(); j++)
            {
                for (int k = 0; k < input[0][0].size(); k++)
                {
                    input[i][j][k] = 1 - (tanh(input[i][j][k])) * (tanh(input[i][j][k]));
                }
            }
        }

        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < input[0].size(); j++)
            {
                for (int k = 0; k < input[0][0].size(); k++)
                {
                    deriv_output[i][j][k] *= input[i][j][k];
                }
            }
        }

        return deriv_output;
    }
};

class Sigmoid : public Activation_Layer
{
public:
    matrix input;
    matrix forward(matrix input)
    {

        this->input = input;

        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < input[0].size(); j++)
            {
                for (int k = 0; k < input[0][0].size(); k++)
                {
                    input[i][j][k] = 1.0 / (1 + exp((-1) * input[i][j][k]));
                }
            }
        }
        return input;
    }

    matrix backward(matrix deriv_output, double learning_rate)
    {

        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < input[0].size(); j++)
            {
                for (int k = 0; k < input[0][0].size(); k++)
                {
                    input[i][j][k] = (1.0 / (1 + exp((-1) * input[i][j][k]))) * (1.0 - 1.0 / (1 + exp((-1) * input[i][j][k])));
                }
            }
        }

        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < input[0].size(); j++)
            {
                for (int k = 0; k < input[0][0].size(); k++)
                {
                    deriv_output[i][j][k] *= input[i][j][k];
                }
            }
        }

        return deriv_output;
    }
};

class Softmax : public Activation_Layer
{
public:
    matrix input;
    matrix output;
    matrix deriv_output;
    matrix forward(matrix input)
    {
        // expecting a matrix i.e a simple vectorized layer
        vector<double> res = input[0][0];

        int size = res.size();
        int i;
        double maxi, sum, constant;

        maxi = -INFINITY;
        for (i = 0; i < size; ++i)
        {
            if (maxi < res[i])
            {
                maxi = res[i];
            }
        }

        sum = 0.0;
        for (i = 0; i < size; ++i)
        {
            sum += exp(res[i] - maxi);
        }

        constant = maxi + log(sum);
        for (i = 0; i < size; ++i)
        {
            res[i] = exp(res[i] - constant);
        }

        output = {{res}};
        return output;
    }

    matrix backward(matrix deriv_output, double learning_rate)
    {
        // expecting a 1d vector

        int reps = deriv_output[0][0].size();

        layer result;
        for (int i = 0; i < reps; i++)
        {
            result.push_back(deriv_output[0][0]);
        }

        layer id = layer(reps, vector<double>(reps, 0));

        for (int i = 0; i < reps; i++)
        {
            for (int j = 0; j < reps; j++)
            {
                if (i == j)
                {
                    id[i][j] = 1;
                }
            }
        }

        layer tmp_transpose = transpose(result);

        for (int i = 0; i < reps; i++)
        {
            for (int j = 0; j < reps; j++)
            {
                id[i][j] = id[i][j] - tmp_transpose[i][j];
                id[i][j] *= result[i][j];
            }
        }

        vector<double> ans = dot_product(id, deriv_output[0][0]);

        deriv_output = {{ans}};

        return deriv_output;
    }
};

/*---------------------------------------------------------------- Reshape Layer --------------------------------------------------------------------*/

class Reshape
{
public:
    int input_height, input_width, input_depth, output_height, output_width, output_depth;

    Reshape(vector<int> input_shape, vector<int> output_shape)
    {
        input_height = input_shape[0];
        input_width = input_shape[1];
        input_depth = input_shape[2];
        output_height = output_shape[0];
        output_width = output_shape[1];
        output_depth = output_shape[2];
    }

    matrix forward(matrix input)
    {
        // essentially being used for converting the Conv2D output to a linear matrix
        matrix output = matrix(output_depth, layer(output_height, vector<double>(output_width, 0)));
        output[0][0].clear();

        for (int depth = 0; depth < input.size(); depth++)
        {

            for (int height = 0; height < input[0].size(); height++)
            {

                for (int width = 0; width < input[0][0].size(); width++)
                {

                    output[0][0].push_back(input[depth][height][width]);
                }
            }
        }

        return output;
    }

    matrix backward(matrix deriv_output, double learning_rate)
    {

        matrix input = matrix(input_depth, layer(input_height, vector<double>(input_width, 0)));

        for (int i = 0; i < deriv_output[0][0].size(); i++)
        {
            double el = deriv_output[0][0][i];

            int depth = i / (input_height * input_width);
            int areal = i % (input_height * input_width);
            int height = areal / (input_height);
            int width = areal % (input_width);

            input[depth][height][width] = el;
        }

        return input;
    }
};

/*---------------------------------------------------------------- Dense Layer ------------------------------------------------------------------------*/

class Dense
{

public:
    int input_shape;
    int output_shape;
    layer weights;
    vector<double> biases;
    matrix output;
    matrix input;

    Dense(int input_shape, int output_shape)
    {
        this->input_shape = input_shape;   // Input connections
        this->output_shape = output_shape; // Output connections
        output = matrix(1, layer(1, vector<double>(output_shape, 0)));
        weights = layer(output_shape, vector<double>(input_shape, 0));
        biases = vector<double>(output_shape, 0);

        random_initialize();
    }

    void random_initialize()
    {

        for (int i = 0; i < weights.size(); i++)
        {
            for (int j = 0; j < weights[0].size(); j++)
            {
                weights[i][j] = (float)rand() / RAND_MAX;
            }
        }

        for (int i = 0; i < biases.size(); i++)
        {
            this->biases[i] = (float)rand() / RAND_MAX;
        }
    }
    void Update_Weights(layer wts_grad, double learning_rate)
    {
        for (int i = 0; i < wts_grad.size(); i++)
        {
            for (int j = 0; j < wts_grad[0].size(); j++)
            {
                this->weights[i][j] -= learning_rate * wts_grad[i][j];
            }
        }
    }

    void Update_bias(matrix deriv_output, double learning_rate)
    {
        for (int i = 0; i < this->biases.size(); i++)
        {
            this->biases[i] -= learning_rate * deriv_output[0][0][i];
        }
    }

    matrix forward(matrix input)
    {
        this->input = input;

        vector<double> op_temp = dot_product(weights, input[0][0]);
        vector<layer> out = {{{}}};

        for (int i = 0; i < biases.size(); i++)
        {
            op_temp[i] += biases[i];
        }
        out[0][0] = op_temp;

        return out;
    }

    matrix backward(matrix deriv_output, double learning_rate)
    {
        // expecting a matrix of size {height = 1, width = some_integer, depth = 1}

        layer weights_gradients, input_gradients;
        layer input1 = {};
        input1.push_back(deriv_output[0][0]);

        input1 = transpose(input1);

        layer input2 = {};
        input2.push_back(this->input[0][0]);

        weights_gradients = matrix_multiplication(input1, input2);
        input_gradients = matrix_multiplication(transpose(this->weights), input1);

        input_gradients = transpose(input_gradients);

        Update_Weights(weights_gradients, learning_rate);
        Update_bias(deriv_output, learning_rate);

        matrix inp_grad = {input_gradients};

        return inp_grad;
    }
};

//------------------------------------------------------------------ Loss --------------------------------------------------------------//

double cross_entropy(vector<double> actual, vector<double> pred)
{

    double val = -1.0;

    int size = actual.size();
    double n = size;

    double exp_1 = 0.0;
    double exp_2 = 0.0;

    for (int i = 0; i < size; i++)
    {
        if (actual[i] == 1 and pred[i] != 0)
            exp_1 += actual[i] * (log(pred[i]));
        else if (actual[i] == pred[i])
            exp_1 += 0;
        else if (pred[i] == 0)
            exp_1 += -100000.0;
    }

    val *= (((exp_1)));

    return val;
}

vector<double> cross_entropy_prime(vector<double> actual, vector<double> pred)
{

    vector<double> result = vector<double>(actual.size());

    double n = result.size();

    for (int i = 0; i < result.size(); i++)
    {
        if (pred[i] != 0 && pred[i] != actual[i])
            result[i] = (-1.0) * (actual[i] / pred[i]);
        else if (pred[i] == actual[i])
            result[i] = -1;
        else if (pred[i] == 0)
            result[i] = (double)-100000.0;
    }

    return result;
}

double bce(vector<double> y_true, vector<double> y_pred)
{
    double loss = 0.0;

    // Loop over the elements in the vectors
    for (int i = 0; i < y_true.size(); i++)
    {
        // Add the binary cross entropy loss for each element to the total loss
        loss += -(y_true[i] * log(y_pred[i]) + (1 - y_true[i]) * log(1 - y_pred[i]));
    }

    // Return the average loss over the elements in the vectors
    return loss / y_true.size();
}

vector<double> bce_prime(vector<double> y_true, vector<double> y_pred)
{
    vector<double> dL(y_true.size());

    // Loop over the elements in the vectors
    for (int i = 0; i < y_true.size(); i++)
    {
        // Handle the case where y_pred is zero or one
        if (y_pred[i] == 0)
        {
            dL[i] = -y_true[i] / 0.0003;
        }
        else if (y_pred[i] == 1)
        {
            dL[i] = (1 - y_true[i]) / 0.0003;
        }
        else
        {
            // Compute the binary cross entropy loss derivative for this element
            dL[i] = -y_true[i] / y_pred[i] + (1 - y_true[i]) / (1 - y_pred[i]);
        }
    }

    return dL;
}

double mean_squared_error(const vector<double> &y_true, const vector<double> &y_pred)
{
    double loss = 0.0;

    // Loop over the elements in the vectors
    for (int i = 0; i < y_true.size(); i++)
    {
        // Add the squared difference between the elements to the total loss
        loss += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }

    // Return the average loss over the elements in the vectors
    return loss;
}

vector<double> mean_squared_error_derivative(const vector<double> &y_true, const vector<double> &y_pred)
{
    vector<double> dL(y_true.size());

    // Loop over the elements in the vectors
    for (int i = 0; i < y_true.size(); i++)
    {
        // Compute the mean squared error loss derivative for this element
        dL[i] = -2 * (y_true[i] - y_pred[i]);
    }

    return dL;
}