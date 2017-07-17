#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////
// The BMCI Class
////////////////////////////////////////////////////////////////////////////////
template <typename TFloat>
class BMCI {
public:
    BMCI(TFloat *Y, TFloat *x, TFloat *s, size_t m, size_t n) : m_(m), n_(n), Y_(*Y), x_(x), s_(s) {}
    BMCI(const BMCI &)  = default;
    BMCI(      BMCI &&) = default;
    BMCI & operator=(const BMCI &&) = default;
    BMCI & operator=(      BMCI &&) = default;
    ~BMCI() = default;

    void expectation(TFloat *x_hat,  const TFloat *Y, size_t m);

private:
    size_t m_, n_;
    TFloat *Y_, *x_, *s_;
};

template <typename TFloat>
void BMCI<TFloat>::expectation(TFloat *x_hat, const TFloat *Y, size_t m)
{
    TFloat *dY = new TFloat[n_];
    for (size_t i = 0; i < m; i++) {
        x_hat[i] = 0.0;
        TFloat p_sum(0.0);
        size_t Y_i_ind(0);
        for (size_t j = 0; j < m_; j++) {
            TFloat p(0.0), dy(0.0), dySdy(0.0);
            size_t Y_j_ind(0);
            for (size_t k = 0; j < n_; j++) {
                dy = Y_[Y_j_ind + k] - Y[Y_i_ind + k];
                dySdy += dy * dy / s_[k];
            }
            p = exp(-dySdy);
            p_sum += p;
            x_hat[i] += p * x_[j];
            Y_j_ind += n_;
        }
        Y_i_ind += n_;
    }
}

////////////////////////////////////////////////////////////////////////////////
// The Python Interface
////////////////////////////////////////////////////////////////////////////////
#include <Python.h>

extern "C" {

    static PyObject*
    bmci_expectation(PyObject *self, PyObject *args)
    {
        PyObject *Y_db_array, *x_db_array, *s_array, *Y_array;
        if (!PyArg_ParseTuple(args, "z", Y_db_array, "z", x_db_array, "z", s_array, "z", Y_array )) {
            return NULL;
        }
        size_t m_db(0), n(0), m(0);
        m_db = PyArray_DIMS(Y_db_array)[0];
        n    = PyArray_DIMS(Y_db_array)[1];
        m    = PyArray_DIMS(Y_array)[0];

        BMCI<double> bmci(PyArray_DATA(Y_db_array), PyArray_DATA(x_db_array), PyArray_DATA(s_array), m_db, n);

        np_intp *dims = new np_intp[2];
        np_intp[0] = m;
        np_intp[1] = 1;
        PyObject *x = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
        bmci.expectation(PyArray_Data(x), PyArray_Data(Y_array), m);

        return x;
    }

    static PyMethodDef methods[] = {
        {"bmci_expectation",  bmci_expectation, METH_VARARGS,
         "Compute expectation value of the retrieval using BMCI"},
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
    PyInit_bmci(void)
    {
        return PyModule_Create(&bmci_module);
    }
}
