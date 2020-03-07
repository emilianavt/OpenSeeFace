// SVMWrapper.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"

#include <cmath>
#include "svm.h"

using namespace std;

extern "C" void __cdecl print_log(const char *str) {
    return;
}

struct SVMWrapper {
    struct svm_model *model;
    int cols;
    int classes;
    double *means;
    double *sdevs;
    struct svm_parameter param;
    struct svm_problem *problem;
};

typedef struct svm_node svmnode;

svmnode **make_svmnode_array(float features[], int rows, int cols, double *means, double *sdevs, int rescale) {
    svmnode **x = new svmnode*[rows];
    svmnode *nodes = new svmnode[rows * (cols + 1)];

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
        x[r] = &(nodes[r * (cols + 1)]);
        int idx = 0;
        for (int c = 0; c < cols; c++) {
            x[r][c].index = c + 1;
            x[r][c].value = ((double)features[r * cols + c] - means[c]) * sdevs[c];
        }
        x[r][cols].index = -1;
    }
    return x;
}

void make_problem(struct svm_problem *problem, float features[], float labels[], int rows, int cols, double *means, double *sdevs) {
    problem->l = rows;
    problem->y = new double[rows];
    for (int i = 0; i < rows; i++)
        problem->y[i] = (double)labels[i];
    problem->x = make_svmnode_array(features, rows, cols, means, sdevs, 1);
}

void destroy_svmnode_array(svmnode **x) {
    delete x[0];
    delete x;
}

void destroy_problem(struct svm_problem *problem) {
    delete problem->y;
    destroy_svmnode_array(problem->x);
}

extern "C" {
    __declspec(dllexport) void *__cdecl trainModel(float features[], float labels[], float weights[], int rows, int cols, int classes, int probability, float C) {
        if (!(rows > 0 && cols > 0))
            return NULL;

        svm_set_print_string_function(print_log);

        struct SVMWrapper *sw = new SVMWrapper();
        sw->cols = cols;
        sw->classes = classes;
        sw->means = new double[cols];
        sw->sdevs = new double[cols];

        sw->param.svm_type = C_SVC;
        sw->param.kernel_type = RBF;
        sw->param.degree = 3;
        sw->param.gamma = 1.0 / (double)cols;
        sw->param.coef0 = 0;
        sw->param.nu = 0.5;
        sw->param.cache_size = 100;
        sw->param.C = C;
        sw->param.eps = 0.001;
        sw->param.p = 0.1;
        sw->param.shrinking = 1;
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

        sw->problem = new struct svm_problem();
        make_problem(sw->problem, features, labels, rows, cols, sw->means, sw->sdevs);
        sw->model = svm_train(sw->problem, &sw->param);

        if (weights != NULL) {
            delete sw->param.weight_label;
            delete sw->param.weight;
        }

        return sw;
    }

    __declspec(dllexport) void __cdecl predict(void *ptr, float features[], float predictions[], double probabilities[], int rows) {
        struct SVMWrapper *sw = (struct SVMWrapper *)ptr;
        int cols = sw->cols;

        svmnode **x = make_svmnode_array(features, rows, cols, sw->means, sw->sdevs, 0);
        for (int r = 0; r < rows; r++)
            predictions[r] = (float)svm_predict_probability(sw->model, x[r], &(probabilities[r * sw->classes]));
        destroy_svmnode_array(x);

        return;
    }

    __declspec(dllexport) void *__cdecl loadModel(char *filename, int cols, int classes, double means[], double sdevs[]) {
        struct SVMWrapper *sw = new SVMWrapper();
        sw->cols = cols;
        sw->model = svm_load_model(filename);
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
        sw->model = svm_load_model_string(modelString);
        sw->classes = classes;
        sw->means = new double[cols];
        sw->sdevs = new double[cols];
        for (int c = 0; c < cols; c++) {
            sw->means[c] = means[c];
            sw->sdevs[c] = sdevs[c];
        }
        sw->problem = NULL;
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

    __declspec(dllexport) char *__cdecl saveModelString (void *ptr) {
        struct SVMWrapper *sw = (struct SVMWrapper *)ptr;
        return svm_save_model_string(sw->model);
    }

    __declspec(dllexport) int __cdecl saveModel(void *ptr, char *filename) {
        struct SVMWrapper *sw = (struct SVMWrapper *)ptr;
        return svm_save_model(filename, sw->model);
    }

    __declspec(dllexport) void __cdecl destroyModel(void *ptr) {
        struct SVMWrapper *sw = (struct SVMWrapper *)ptr;
        delete sw->means;
        delete sw->sdevs;
        svm_free_and_destroy_model(&sw->model);
        if (sw->problem != NULL) {
            destroy_problem(sw->problem);
            delete sw->problem;
        }
        delete sw;
    }
}