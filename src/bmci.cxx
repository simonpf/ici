#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////
// The BMCI Class
////////////////////////////////////////////////////////////////////////////////
template <typename TFloat>
class BMCI {
public:
    BMCI(const double *Y, const double *x, const double *s, size_t m, size_t n)
        : m_(m), n_(n), Y_(new TFloat[m * n]), x_(new TFloat[m]), s_(new TFloat[n]),
          hist_min_(std::numeric_limits<TFloat>::max()), hist_max_(0.0)
    {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                Y_[i * n + j] = static_cast<TFloat>(Y[i * n + j]);
            }
            x_[i] = static_cast<TFloat>(x[i]);

            hist_min_ = (x[i] != 0.0) ? std::min(hist_min_, x_[i]) : hist_min_;
            hist_max_ = std::max(hist_max_, x_[i]);
        }
        hist_max_ *= (1.0 + 1e-6);

        for (size_t i = 0; i < n; ++i) {
            s_[i] = static_cast<TFloat>(s[i]);
        }
    }

    BMCI(const BMCI &)  = default;
    BMCI(      BMCI &&) = default;
    BMCI & operator=(const BMCI &)  = default;
    BMCI & operator=(      BMCI &&) = default;

    ~BMCI()
    {
        delete Y_;
        delete x_;
        delete s_;
    };

    void expectation(TFloat *x_hat,  const TFloat *Y, size_t m);
    void pdf(TFloat *x_hat, TFloat *hist,  const TFloat *Y, size_t m, size_t n_bins);

private:

    inline size_t get_bin(TFloat x, TFloat hist_min, TFloat d_hist);
    size_t m_, n_;
    TFloat *Y_, *x_, *s_;

    TFloat hist_min_, hist_max_;
};

template <typename TFloat>
size_t BMCI<TFloat>::get_bin(TFloat x, TFloat hist_min, TFloat d_hist)
{
    if (x == 0.0) {
        return 0;
    } else {
        TFloat i_f = (x - hist_min) / d_hist;
        return 1 + std::trunc(i_f);
    }
}

template <typename TFloat>
void BMCI<TFloat>::expectation(TFloat *x_hat, const TFloat *Y, size_t m)
{
    TFloat *dY = new TFloat[n_];
    size_t Y_i_ind(0);
    for (size_t i = 0; i < m; i++) {
        x_hat[i] = 0.0;
        TFloat p_sum(0.0);
        size_t Y_j_ind(0);
        for (size_t j = 0; j < m_; j++) {
            TFloat p(0.0), dy(0.0), dySdy(0.0);
            for (size_t k = 0; k < n_; k++) {
                dy = Y_[Y_j_ind + k] - Y[Y_i_ind + k];
                dySdy += dy * dy / s_[k];
            }
            p = exp(-dySdy);
            p_sum += p;
            x_hat[i] += p * x_[j];
            Y_j_ind += n_;
        }
        x_hat[i] /= p_sum;
        Y_i_ind += n_;
    }
}

template <typename TFloat>
void BMCI<TFloat>::pdf(TFloat *x_hat, TFloat *hist, const TFloat *Y, size_t m, size_t n_bins)
{
    TFloat *dY = new TFloat[n_];
    TFloat hist_min_log = log(hist_min_);
    TFloat d_hist = (log(hist_max_) - hist_min_log) / (n_bins - 1);
    size_t Y_i_ind(0), hist_ind(0);

    for (size_t i = 0; i < m; i++) {
        x_hat[i] = 0.0;
        TFloat p_sum(0.0);
        size_t Y_j_ind(0);
        for (size_t j = 0; j < m_; j++) {
            TFloat p(0.0), dy(0.0), dySdy(0.0);
            for (size_t k = 0; k < n_; k++) {
                dy = Y_[Y_j_ind + k] - Y[Y_i_ind + k];
                dySdy += dy * dy / s_[k];
            }
            p = exp(-dySdy);
            p_sum += p;
            x_hat[i] += p * x_[j];
            Y_j_ind += n_;

            // Add p to histogram bin.
            TFloat x_log = (x_[j] == 0.0) ? 0.0 : log(x_[j]);
            hist[hist_ind + get_bin(x_log, hist_min_log, d_hist)] += p;

        }
        x_hat[i] /= p_sum;
        Y_i_ind += n_;
        hist_ind += n_bins;
    }
}

////////////////////////////////////////////////////////////////////////////////
// The Python Interface
////////////////////////////////////////////////////////////////////////////////
#include <Python.h>
#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>

extern "C" {

    BMCI<float>  *bmci_float;
    BMCI<double> *bmci_double;

    static PyObject *
    bmci_initialize(PyObject *self, PyObject *args)
    {
        // Parse Arguments.
        PyObject *Y_array, *x_array, *s_array;
        Py_ssize_t prec;

        if (!PyArg_ParseTuple(args, "OOOn", &Y_array, &x_array, &s_array, &prec)) {
            return NULL;
        }
        size_t m(0), n(0);
        m = (size_t) PyArray_DIMS(Y_array)[0];
        n = (size_t) PyArray_DIMS(Y_array)[1];

        if (prec == 32) {
            bmci_float = static_cast<BMCI<float>*>(
                new BMCI<float>(static_cast<double*>(PyArray_DATA(Y_array)),
                                static_cast<double*>(PyArray_DATA(x_array)),
                                static_cast<double*>(PyArray_DATA(s_array)),
                                m, n));
        } else if (prec == 64) {
            bmci_double = static_cast<BMCI<double>*>(
                new BMCI<double>(static_cast<double*>(PyArray_DATA(Y_array)),
                                 static_cast<double*>(PyArray_DATA(x_array)),
                                 static_cast<double*>(PyArray_DATA(s_array)),
                                 m, n));
        } else {
            return NULL;
        }

        Py_RETURN_NONE;
    }

    static PyObject *
    bmci_finalize(PyObject */*self*/, PyObject */*args*/)
    {
        if (bmci_float) {
            delete bmci_float;
        }
        if (bmci_double) {
            delete bmci_double;
        }
        Py_RETURN_NONE;
    }

    static PyObject *
    bmci_expectation(PyObject *self, PyObject *args)
    {
        // Parse Arguments.
        PyObject *Y_array;
        if (!PyArg_ParseTuple(args, "O", &Y_array)) {
            return NULL;
        }
        size_t m(0);
        m    = (size_t) PyArray_DIMS(Y_array)[0];

        npy_intp *dims = new npy_intp[2];
        dims[0] = m;
        dims[1] = 1;

        // Run BMCI depending on type.
        PyObject *x;
        if ((PyArray_TYPE(Y_array) == NPY_FLOAT32) && bmci_float) {
            x = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
            bmci_float->expectation(static_cast<float*>(PyArray_DATA(x)),
                                    static_cast<float*>(PyArray_DATA(Y_array)),
                                    m);
        } else if ((PyArray_TYPE(Y_array) == NPY_FLOAT64) && bmci_double) {
            x = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
            bmci_double->expectation(static_cast<double*>(PyArray_DATA(x)),
                                     static_cast<double*>(PyArray_DATA(Y_array)),
                                     m);
        } else {
            return NULL;
        }

        // Clean up.
        delete[] dims;
        return x;
    }

    static PyObject *
    bmci_pdf(PyObject *self, PyObject *args)
    {
        // Parse Arguments.
        PyObject *Y_array;
        Py_ssize_t n_bins;
        if (!PyArg_ParseTuple(args, "On", &Y_array, &n_bins)) {
            return NULL;
        }
        size_t m(0);
        m    = (size_t) PyArray_DIMS(Y_array)[0];

        // Run BMCI depending on type.
        npy_intp *x_dims = new npy_intp[2];
        x_dims[0] = m;
        x_dims[1] = 1;
        npy_intp *hist_dims = new npy_intp[2];
        hist_dims[0] = m;
        hist_dims[1] = n_bins;

        PyObject *x, *hist;
        if ((PyArray_TYPE(Y_array) == NPY_FLOAT32) && bmci_float) {
            x = PyArray_SimpleNew(2, x_dims, NPY_FLOAT32);
            hist = PyArray_ZEROS(2, hist_dims, NPY_FLOAT32, 0);
            bmci_float->pdf(static_cast<float*>(PyArray_DATA(x)),
                            static_cast<float*>(PyArray_DATA(hist)),
                            static_cast<float*>(PyArray_DATA(Y_array)),
                            m, n_bins);
        } else if ((PyArray_TYPE(Y_array) == NPY_FLOAT64) && bmci_double) {
            x = PyArray_SimpleNew(2, x_dims, NPY_FLOAT64);
            hist = PyArray_ZEROS(2, hist_dims, NPY_FLOAT64, 0);
            bmci_double->pdf(static_cast<double*>(PyArray_DATA(x)),
                             static_cast<double*>(PyArray_DATA(hist)),
                             static_cast<double*>(PyArray_DATA(Y_array)),
                             m, n_bins);
        } else {
            return NULL;
        }

        // Zip arrays.
        PyObject *pair = PyTuple_New(2);
        PyTuple_SetItem(pair, 0, x);
        PyTuple_SetItem(pair, 1, hist);

        // Clean up.
        delete[] x_dims;
        delete[] hist_dims;

        // Done.
        return pair;
    }

    static PyMethodDef methods[] = {
        {"bmci_initialize",  bmci_initialize, METH_VARARGS, "Initialize bmci object."},
        {"bmci_finalize",  bmci_finalize, METH_NOARGS, "Finalize bmci object."},
        {"bmci_expectation",  bmci_expectation, METH_VARARGS, "Compute expectation value of the retrieval using BMCI"},
        {"bmci_pdf",  bmci_pdf, METH_VARARGS, "Compute expectation and PDF of the posterior value of the retrieval using BMCI"},
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };

    static struct PyModuleDef bmci_module = {
        PyModuleDef_HEAD_INIT,
        "bmci",   /* name of module */
        NULL,
        -1,
        methods
    };

    PyMODINIT_FUNC
    PyInit_bmcixx(void)
    {
        import_array();
        return PyModule_Create(&bmci_module);
    }
}
