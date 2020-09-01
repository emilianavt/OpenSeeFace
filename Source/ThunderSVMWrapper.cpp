// ThunderSVMWrapper.cpp : Defines the exported functions for the DLL.
//

#include "pch.h"
#include "framework.h"
#include "ThunderSVMWrapper.h"
#include "thundersvm/thundersvm.h"
#include "thundersvm/model/svmmodel.h"
#include "thundersvm/model/svc.h"
#include <cmath>

using namespace std;

extern "C" void __cdecl print_log(const char *str) {
    return;
}

struct problem {
    int r;
    int c;
    DataSet::node2d x;
    vector<double> y;
    DataSet dataset;
};

struct SVMWrapper {
    SVC *model;
    int cols;
    int classes;
    double *means;
    double *sdevs;
    SvmParam param;
    struct problem problem;
};

typedef struct svm_node svmnode;

void make_svmnode_array(DataSet::node2d &dataset, float features[], int rows, int cols, double *means, double *sdevs, int rescale) {
    double row_factor = 1.0 / (double)rows;
    if (rescale) {
        for (int c = 0; c < cols; c++) {
            means[c] = 0;
            sdevs[c] = 0;
        }
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                means[c] += (double)features[r * cols + c];
            }
        }
        for (int c = 0; c < cols; c++) {
            means[c] *= row_factor;
        }
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double sq_diff = ((double)features[r * cols + c] - means[c]);
                sq_diff *= sq_diff;
                sdevs[c] += sq_diff;
            }
        }
        for (int c = 0; c < cols; c++) {
            sdevs[c] *= row_factor;
            sdevs[c] = sqrt(sdevs[c]);
            sdevs[c] = 1.0 / sdevs[c];
        }
    }

    for (int r = 0; r < rows; r++) {
        dataset[r] = vector<DataSet::node>(cols);
        int idx = 0;
        for (int c = 0; c < cols; c++) {
            dataset[r][c].index = c + 1;
            dataset[r][c].value = (float)((features[r * cols + c] - means[c]) * sdevs[c]);
        }
    }
}

void make_problem(struct problem &problem, float features[], float labels[], int rows, int cols, double *means, double *sdevs) {
    problem.r = rows;
    problem.c = cols;
    problem.y = vector<double>(rows);
    for (int i = 0; i < rows; i++)
        problem.y[i] = (double)labels[i];
    problem.x = DataSet::node2d(rows);
    make_svmnode_array(problem.x, features, rows, cols, means, sdevs, 1);
    problem.dataset = DataSet(problem.x, cols, problem.y);
}

void destroy_problem(struct problem &problem) {
    problem.y.clear();
    problem.x.clear();
}

extern "C" {
    __declspec(dllexport) void *__cdecl trainModel(float features[], float labels[], float weights[], int rows, int cols, int classes, int probability, float C) {
        if (!(rows > 0 && cols > 0))
            return NULL;

        //svm_set_print_string_function(print_log);

        struct SVMWrapper *sw = new SVMWrapper();
        sw->cols = cols;
        sw->classes = classes;
        sw->means = new double[cols];
        sw->sdevs = new double[cols];

        sw->param.svm_type = SvmParam::C_SVC;
        sw->param.kernel_type = SvmParam::RBF;
        sw->param.degree = 3;
        sw->param.gamma = 1.0 / (double)cols;
        sw->param.coef0 = 0;
        sw->param.nu = 0.5;
        sw->param.C = C;
        sw->param.epsilon = 0.001;
        sw->param.p = 0.1;
        sw->param.probability = probability;
        sw->param.nr_weight = 0;
        sw->param.weight_label = NULL;
        sw->param.weight = NULL;

        if (weights != NULL) {
            sw->param.nr_weight = classes;
            sw->param.weight_label = new int[classes];
            sw->param.weight = new double[classes];
            for (int i = 0; i < classes; i++) {
                sw->param.weight_label[i] = i;
                sw->param.weight[i] = (double)weights[i];
            }
        }

        make_problem(sw->problem, features, labels, rows, cols, sw->means, sw->sdevs);
        sw->model = new SVC();//svm_train(sw->problem, &sw->param);
        sw->model->train(sw->problem.dataset, sw->param);

        if (weights != NULL) {
            delete sw->param.weight_label;
            delete sw->param.weight;
        }

        return sw;
    }

    __declspec(dllexport) void __cdecl predict(void *ptr, float features[], float predictions[], double probabilities[], int rows) {
        struct SVMWrapper *sw = (struct SVMWrapper *)ptr;
        int cols = sw->cols;

        DataSet::node2d instances = DataSet::node2d(rows);
        make_svmnode_array(instances, features, rows, sw->cols, sw->means, sw->sdevs, 0);
        vector<double> predicted = sw->model->predict(instances, -1);
        instances.clear();
        for (int r = 0; r < rows; r++)
            predictions[r] = (float)predicted[r];
        if (probabilities != NULL && sw->param.probability) {
            vector<float> probs = sw->model->get_prob_predict();
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < sw->classes; c++)
                    probabilities[c + r * sw->classes] = probs[c + r * sw->classes];
        }
        predicted.clear();

        return;
    }

    __declspec(dllexport) void *__cdecl loadModel(char *filename, int cols, int classes, double means[], double sdevs[]) {
        struct SVMWrapper *sw = new SVMWrapper();
        sw->cols = cols;
        sw->model = new SVC();
        sw->model->load_from_file(filename);
        sw->classes = classes;
        sw->means = new double[cols];
        sw->sdevs = new double[cols];
        for (int c = 0; c < cols; c++) {
            sw->means[c] = means[c];
            sw->sdevs[c] = sdevs[c];
        }
        return sw;
    }

    __declspec(dllexport) void *__cdecl loadModelString(char *modelString, int cols, int classes, double means[], double sdevs[]) {
        struct SVMWrapper *sw = new SVMWrapper();
        sw->cols = cols;
        sw->model = new SVC();
        sw->model->load_from_string(modelString);
        sw->classes = classes;
        sw->means = new double[cols];
        sw->sdevs = new double[cols];
        for (int c = 0; c < cols; c++) {
            sw->means[c] = means[c];
            sw->sdevs[c] = sdevs[c];
        }
        return sw;
    }

    __declspec(dllexport) void __cdecl getScales(void *ptr, double means[], double sdevs[]) {
        struct SVMWrapper *sw = (struct SVMWrapper *)ptr;
        int cols = sw->cols;
        for (int c = 0; c < cols; c++) {
            means[c] = sw->means[c];
            sdevs[c] = sw->sdevs[c];
        }
    }

    __declspec(dllexport) const char *__cdecl saveModelString(void *ptr) {
        struct SVMWrapper *sw = (struct SVMWrapper *)ptr;
        string model = sw->model->save_to_string();
        return _strdup(model.c_str());
    }

    __declspec(dllexport) int __cdecl saveModel(void *ptr, char *filename) {
        struct SVMWrapper *sw = (struct SVMWrapper *)ptr;
        sw->model->save_to_file(string(filename));
        return 1;
    }

    __declspec(dllexport) void __cdecl destroyModel(void *ptr) {
        struct SVMWrapper *sw = (struct SVMWrapper *)ptr;
        delete sw->means;
        delete sw->sdevs;
        delete sw->model;
        destroy_problem(sw->problem);
        delete sw;
    }
}