#include <math.h>
#include <Python.h>
#include "knapsack.h"
#include "bruteforce.h"
#include "problem_def.h"

static PyObject* gerrymandering_maximize_probability(PyObject* self, PyObject* args, PyObject* kwargs) {
    int m, b, budget, target;
    PyObject* p_s_list;
    const char* solver = "backtracking";
    int fast = 0;

    static const char* kwlist[] = {"m", "b", "budget", "target", "p_s", "solver", "fast", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiiO|si", (char**)kwlist,
                                     &m, &b, &budget, &target, &p_s_list, &solver, &fast)) {
        return NULL;
    }

    if (!PyList_Check(p_s_list)) {
        PyErr_SetString(PyExc_TypeError, "p_s must be a list");
        return NULL;
    }

    GerrymanderingInstance gi;
    gi.m = m;
    gi.b = b;
    gi.budget = budget;
    gi.target = target;

    Py_ssize_t p_s_size = PyList_Size(p_s_list);
    gi.p_s.resize(p_s_size);
    for (Py_ssize_t i = 0; i < p_s_size; ++i) {
        PyObject* item = PyList_GetItem(p_s_list, i);
        if (!PyFloat_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "p_s list items must be floats");
            return NULL;
        }
        gi.p_s[i] = PyFloat_AsDouble(item);
    }

    MaximizationResult result;
    if (strcmp(solver, "knapsack") == 0) {
        result = knapsack::maximize_probability(gi, fast);
    } else {
        result = bruteforce::maximize_probability(gi);
    }

    PyObject* result_dict = PyDict_New();
    PyDict_SetItemString(result_dict, "phi_c", PyFloat_FromDouble(result.phi_c));
    PyDict_SetItemString(result_dict, "log_phi_c", PyFloat_FromDouble(result.phi_c ? std::log(result.phi_c) : std::nan("")));

    PyObject* choices_list = PyList_New(result.choices.size());
    for (size_t i = 0; i < result.choices.size(); ++i) {
        PyList_SetItem(choices_list, i, PyLong_FromLong(result.choices[i]));
    }
    PyDict_SetItemString(result_dict, "choices", choices_list);

    return result_dict;
}

static PyMethodDef GerrymanderingMethods[] = {
    {"maximize_probability", (PyCFunction)gerrymandering_maximize_probability, METH_VARARGS | METH_KEYWORDS,
     "Maximize the probability of winning the election."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gerrymanderingmodule = {
    PyModuleDef_HEAD_INIT,
    "gerrymandering_ext",
    "Python interface for the gerrymandering C++ library",
    -1,
    GerrymanderingMethods
};

PyMODINIT_FUNC PyInit_gerrymandering_ext(void) {
    return PyModule_Create(&gerrymanderingmodule);
}
