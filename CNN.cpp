#include "Core.h"

Convolutional l1 = Convolutional({28, 28, 1}, 5, 3, false);
ReLU l2 = ReLU();
Convolutional l3 = Convolutional({26, 26, 5}, 5, 3, false);
ReLU l31 = ReLU();
Reshape l32 = Reshape({24, 24, 5}, {1, 24 * 24 * 5, 1});
Dense l4 = Dense(5 * 24 * 24, 100);
Sigmoid l5 = Sigmoid();
Dense l6 = Dense(100, 10);
Tanh l7 = Tanh();
Dense l8 = Dense(10, 10);
Sigmoid l9 = Sigmoid();
Softmax l11 = Softmax();

int correct = 0;
int epoch_globe = 0;

vector<matrix> data_loader()
{
    string fname = "mnist_train.csv";

    vector<vector<string>> content;
    vector<string> row;
    string line, word;

    fstream file(fname, ios::in);
    if (file.is_open())
    {
        while (getline(file, line))
        {
            row.clear();

            stringstream str(line);

            while (getline(str, word, ','))
                row.push_back(word);
            content.push_back(row);
        }
    }
    else
        cout << "Error opening the file\n";

    vector<double> labels = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    layer set_labels;
    vector<matrix> images_set;
    layer mat_image = layer(28, vector<double>(28, 0));
    layer temp = mat_image;
    vector<matrix> data;

    for (int i = 1; i <= 15000; i++)
    {

        int ly = stoi(content[i][0]);
        if (true)
        {
            labels[ly] = 1;
            set_labels.push_back(labels);
            labels[ly] = 0;

            for (int j = 1; j <= 784; j++)
            {

                mat_image[(j - 1) / 28][(j - 1) % 28] = (double)stoi(content[i][j]);
            }
            images_set.push_back({mat_image});
            mat_image = temp;
        }
    }

    images_set.push_back({set_labels});

    return images_set;
}

double one_iter(matrix t1, vector<double> label, int epoch)
{

    matrix temp1 = l1.forward(t1);
    t1 = temp1;
    temp1 = l2.forward(t1);
    t1 = temp1;
    temp1 = l3.forward(t1);
    t1 = temp1;
    temp1 = l31.forward(t1);
    t1 = temp1;
    temp1 = l32.forward(t1);
    t1 = temp1;
    temp1 = l4.forward(t1);
    t1 = temp1;
    temp1 = l5.forward(t1);
    t1 = temp1;
    temp1 = l6.forward(t1);
    t1 = temp1;
    temp1 = l7.forward(t1);
    t1 = temp1;
    temp1 = l8.forward(t1);
    t1 = temp1;
    temp1 = l9.forward(t1);
    t1 = temp1;
    temp1 = l11.forward(t1);
    t1 = temp1;

    vector<double> mem = t1[0][0];

    int maximumelemnt_index = max_element(mem.begin(), mem.end()) - mem.begin();

    int maximumelemnt_index_label = max_element(label.begin(), label.end()) - label.begin();

    if (maximumelemnt_index == maximumelemnt_index_label)
    {
        correct++;
    }

    for (int it = 0; it < mem.size(); it++)
    {

        cout << mem[it] << " ";
    }
    cout << endl;

    double loss_val = cross_entropy(label, mem);

    for (auto it : label)
    {
        cout << it << " ";
    }
    cout << endl;

    vector<double> loss_der = cross_entropy_prime(label, mem);

    matrix t2 = {{loss_der}};
    cout << " Loss : " << loss_val << " ";
    cout << endl;

    if (isnan(loss_val) or loss_val > 50)
        return 10000;

    // Backward

    matrix temp = l11.backward(t2, 0.0013);
    t2 = temp;
    temp = l9.backward(t2, 0.0013);
    t2 = temp;
    temp = l8.backward(t2, 0.0013);
    t2 = temp;
    temp = l7.backward(t2, 0.0013);
    t2 = temp;
    temp = l6.backward(t2, 0.0013);
    t2 = temp;
    temp = l5.backward(t2, 0.0013);
    t2 = temp;
    temp = l4.backward(t2, 0.0013);
    t2 = temp;
    temp = l32.backward(t2, 0.0013);
    t2 = temp;
    temp = l31.backward(t2, 0.0013);
    t2 = temp;
    temp = l3.backward(t2, 0.0013);
    t2 = temp;
    temp = l2.backward(t2, 0.0013);
    t2 = temp;
    temp = l1.backward(t2, 0.0013);
    t2 = temp;

    cout << endl;

    return loss_val;
}

int main()
{
    vector<matrix> images_set = data_loader();
    int epochs = 5;
    double loss_epoch;

    for (int j = 0; j < epochs; j++)
    {
        int iter = j;
        for (int i = 0; i < images_set.size() - 1; i++)
        {
            matrix t1 = images_set[i];
            vector<double> label = images_set[images_set.size() - 1][0][i];

            loss_epoch = one_iter(t1, label, iter);
        }
        cout << loss_epoch << endl;
        cout << correct << endl;
        correct = 0;
        epoch_globe++;
    }
}